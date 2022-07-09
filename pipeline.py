import translators as ts
from pyspark.sql import SparkSession
from pyspark.sql.functions import first, col, split, udf, initcap

class CarsPipeline:
    """
    Class
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        self.spark = SparkSession.builder.getOrCreate()

    def execute(self):
        """
        Pipeline execute method
        """
        self.pre_process()
        self.normalise()
        self.extract()
        self.integrate()

    def pre_process(self):
        """
        Pre-processing method
        """
        self.spark.read\
            .json('supplier_car.json', encoding='utf-8')\
            .groupBy(['ID', 'MakeText', 'ModelText', 'ModelTypeText', 'TypeName', 'TypeNameFull'])\
            .pivot('Attribute Names')\
            .agg(first('Attribute Values'))\
            .write.csv('pre_processed_data', sep='\t', header=True, encoding='utf-8', mode='overwrite')

    def normalise(self):
        """
        Normalisation method
        """
        norm_df = self.spark.read\
            .csv('pre_processed_data', sep='\t', header=True, encoding='utf-8', inferSchema=True)

        #google translator udf
        udf_translate = udf(lambda x:ts.google(x, from_language='de'))

        color_map = norm_df.select('BodyColorText').distinct()\
                        .withColumn('BodyColorTextShort', split(col('BodyColorText'), ' ').getItem(0))\
                        .withColumn('BodyColorTextEng', initcap(udf_translate(col('BodyColorTextShort'))))

        cond_map = norm_df.select('ConditionTypeText').distinct()\
                        .withColumn('ConditionTypeTextEng', initcap(udf_translate(col('ConditionTypeText'))))

        norm_df = norm_df.join(color_map, ['BodyColorText'])\
                         .join(cond_map, ['ConditionTypeText'])

        norm_df = norm_df.withColumn('MakeTextNorm', initcap('MakeText'))
        norm_df.write.csv('normalised_data', sep='\t', header=True, encoding='utf-8', mode='overwrite')

    def extract(self):
        """
        Exctraction method
        """
        ext_df = self.spark.read\
            .csv('normalised_data', sep='\t', header=True, encoding='utf-8', inferSchema=True)
        ext_df = ext_df.withColumn('extracted-value-ConsumptionTotalText',\
            split(col('ConsumptionTotalText'), ' ').getItem(0))\
                .withColumn('extracted-unit-ConsumptionTotalText',\
            split(col('ConsumptionTotalText'), ' ').getItem(1))
        ext_df.write.csv('extracted_data', sep='\t', header=True, encoding='utf-8', mode='overwrite')

    def integrate(self):
        """
        Integration method
        """
        col_map = {'BodyColorTextEng': 'color',
                    'ConditionTypeTextEng':'condition',
                    'City':'city',
                    'MakeTextNorm':'make',
                    'ModelText':'model',
                    'TypeName':'model_variant'}

        int_df = self.spark.read\
            .csv('extracted_data', sep='\t', header=True, encoding='utf-8', inferSchema=True)

        for k, v in col_map.items():
            int_df = int_df.withColumnRenamed(k, v)

        int_df.select('make', 'model', 'model_variant', 'color', 'condition', 'city')\
            .write.csv('integrated_data', sep='\t', header=True, encoding='utf-8', mode='overwrite')


c = CarsPipeline()
c.execute()