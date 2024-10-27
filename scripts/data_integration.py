import sys
import boto3
from botocore.exceptions import ClientError
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, explode, lit, concat
from awsglue.dynamicframe import DynamicFrame

# Get Job Arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Load the inferred schema from Glue Data Catalog
catalog_table = glueContext.create_dynamic_frame.from_catalog(database="dataintegration", table_name="schema_csv")
schema_df = catalog_table.toDF()
final_schema = schema_df.schema

schema_df.printSchema()

# Load Data from JSON Files in S3
raw_users_df = spark.read.option("multiline", "true").json("s3://data-integration-demo/users.json")
raw_orders_df = spark.read.option("multiline", "true").json("s3://data-integration-demo/orders.json")
raw_products_df = spark.read.option("multiline", "true").json("s3://data-integration-demo/products.json")

orders_df = raw_orders_df.select(explode(col("orders")).alias("order")).select("order.*")
products_df = raw_products_df.select(explode(col("products")).alias("product")).select("product.*")
users_df = raw_users_df.select(explode(col("users")).alias("user")).select("user.*")

orders_df_flattened = orders_df.withColumn("item", explode(col("items"))).select(
    col("order_id").cast("int"),
    col("customer_id").cast("int"),
    col("order_date"),
    col("total_amount").cast("double"),
    col("item.item_id").alias("item_id"),
    col("item.product_name").alias("item_product_name"),
    col("item.quantity").alias("quantity"),
    col("item.price").alias("item_price")
)

products_df_flattened = products_df.select(
    col("product_id"),
    col("product_name").alias("product_product_name"),
    col("category"),
    col("price").cast("double"),
    col("stock_quantity").cast("int")
)

users_df_flattened = users_df.select(
    col("user_id").cast("int"),
    col("name.first_name").alias("first_name"),
    col("name.last_name").alias("last_name"),
    col("contact.email").alias("email"),
    col("contact.phone").alias("phone"),
    col("address.home.street").alias("home_street"),
    col("address.home.city").alias("home_city"),
    col("address.home.zipcode").alias("home_zipcode"),
    col("address.office.street").alias("office_street"),
    col("address.office.city").alias("office_city"),
    col("address.office.zipcode").alias("office_zipcode")
)

merged_df = orders_df_flattened.join(users_df_flattened, orders_df_flattened["customer_id"] == users_df_flattened["user_id"], "left") \
                               .drop(users_df_flattened["user_id"])

final_df = merged_df.join(products_df_flattened, merged_df["item_product_name"] == products_df_flattened["product_product_name"], "left") \
                    .withColumnRenamed("customer_id", "user_id") \
                    .withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name")))

final_df_with_schema = final_df.select(
    [col(field.name).cast(field.dataType).alias(field.name) for field in final_schema.fields]
)


final_df_with_schema.printSchema()

output_path = "s3://data-integration-demo/merged_data/"
final_df_with_schema.coalesce(1).write.mode("overwrite").json(output_path)

glue_client = boto3.client("glue")

# Define database and table name
database_name = "dataintegration"
table_name = "merged_data_table"

try:
    glue_client.get_table(DatabaseName=database_name, Name=table_name)
    print(f"Table '{table_name}' already exists in Glue. No action needed.")

except ClientError as e:
    if e.response['Error']['Code'] == 'EntityNotFoundException':
        print(f"Table '{table_name}' not found. Creating it in Glue.")
        
        # Define the schema based on final_df_with_schema
        columns = [{"Name": field.name, "Type": field.dataType.simpleString()} for field in final_df_with_schema.schema.fields]

        # Create the table
        glue_client.create_table(
            DatabaseName=database_name,
            TableInput={
                'Name': table_name,
                'StorageDescriptor': {
                    'Columns': columns,
                    'Location': output_path,
                    'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.openx.data.jsonserde.JsonSerDe'
                    }
                },
                'TableType': 'EXTERNAL_TABLE'
            }
        )
        print(f"Table '{table_name}' created in Glue with data stored at {output_path}.")
    else:
        print("An error occurred:", e)


job.commit()
