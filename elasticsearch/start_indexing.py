import pandas as pd
import os.path
from elasticsearch import Elasticsearch
import codecs
import requests

#######################################################################################################################
# Version Check
# python       : 2.7.12
# pandas       : 0.23.4
# Elasticsearch: (For Elasticsearch 6.0 and later, use the major version 6 (6.x.y) of the library, we are using 6.3.2)
#                For more details, see python client for elasticsearch at https://elasticsearch-py.readthedocs.io/en/master/index.html
# codecs       : Compatible with python 2.7.12 
# requests     : 2.20.0
########################################################################################################################

#############
# Functions
#############

# Return full file
def return_path(row):
    #Some implementation here#
    _str = ''
    return _str


###########################
# Elasticsearch Connection
###########################
# IF ES is running on localhost
res = requests.get('http://127.0.0.1:9200')
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])
print('ElasticSearch Response Code (200 is OK): {}'.format(res))

#############################
# URI for MYSQL Database
#############################
db_uri='mysql+pymysql://username:password@mysqlserver:port/database_name'

######################
# Getting table
######################
sql_query = 'select * from database_name.table_name_1'
tab_1 = pd.read_sql_query(sql_query, db_uri)

############
# INDEXING 
############
try:
    # Check if file exists
    for _index, _row in tab_1.iterrows():
        if os.path.isfile(_row['Mount_FilePath']) is True:
            print('INDEXING FILE: %s\n' %(_row['Mount_FilePath']) )
            # index files
            file_to_read = *filename* 
            index_in = *index*
            doc_id = *id*
            
            with codecs.open(file_to_read, 'r', encoding='UTF-16LE') as file_obj:
                _text = file_obj.read()
    
            # Add it to the elasticsearch index using default mapping
            doc = {
                "content": _text
            }
            ###########################
            # Do the indexing
            ###########################
            res = es.index(index=index_in, doc_type='_doc', id=doc_id, refresh='wait_for', body=doc)

except Exception as e:
    print(e)
