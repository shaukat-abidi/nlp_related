#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  es_connection.py
#  
#  Copyright 2018 Syed Shaukat Raza Abidi <abi008@demak-ep>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from elasticsearch import Elasticsearch
import requests

class connection:
	def __init__(self, _host = '127.0.0.1', _port = '9200'):
		self.host = _host
		self.port = _port
		self.address = 'http://' + self.host + ':' + str(self.port)

	def connect(self, logger):
		logger.info('Connecting to : %s' %(self.address) )
		#print('Connecting to : %s' %(self.address) )
		res = requests.get(self.address)
		
		if(res.status_code == 200):
			logger.info('Response Code (200 is OK): {}'.format(res))
			#print('Response Code (200 is OK): {}'.format(res))
		else:
			logger.info('Something went wrong, response code: %d' %(res.status_code) )
			#print('Something went wrong, response code: %d' %(res.status_code) )
		
		# connecting to elastic search node on host
		es = Elasticsearch([{'host': self.host, 'port': self.port}])
		logger.info('Connected to : %s' %(self.address) )
		#print('Connected to : %s' %(self.address) )
		
		return es
