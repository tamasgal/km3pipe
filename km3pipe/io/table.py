from km3pipe.core import Pump, Module, Blob
import pandas as pd
import numpy as np
from astropy.io import fits
import json
import yaml

class TablePump(Pump):
	def configure(self):
		self.filename = self.get('filename')
		self.configname = self.get('configname', default='')
		self.meta = {}
		self.iterator = 0
		self.table = False

		self._read_config()

	def _open_infile(self, configs = False):
		if not configs:
			configs = {"filepath_or_buffer":self.filename, "sep":"\t", "header":[0,1], "infer_datetime_format":True}
		if "filename" in configs:
			configs["filepath_or_buffer"]=configs["filename"]
			del configs["filename"]
		if not "filepath_or_buffer" in configs:
			configs["filepath_or_buffer"] = self.filename
		self.table = pd.read_table(**configs)

	def _read_config(self):
		""" Assuming a yaml-file specifying key-value pairs.
		Needs basic elements 'header' and 'parameters' like
		
			header:
				infokey: first info
				
			parameters:
				key1:
					someattribute: some info
		Parameters for the csv reading can be passed in the key "infile"
		"""
		if self.configname:
			meta = yaml.safe_load(open(self.configname, "r").read())
			if "header" in meta:
				self.meta["header"] = meta["header"]
			if "parameters" in meta:
				self.meta["parameters"] = meta["parameters"]
			if "infile" in meta:
				self._open_infile(meta["infile"])
			else:
				self._open_infile()
		self.expose(self.meta, "TableMetaData")

	def process(self, blob):
		blob = Blob()

		entry = self.table.iloc[self.iterator]
		for key, value in entry.iteritems():
			thiskey = key
			if type(key) is tuple:
				thiskey = key[0]
			blob.setdefault(thiskey, value)
		self.iterator +=1

		if self.iterator == len(self.table):
			raise StopIteration
		return blob

class TableSink(Module):
	def configure(self):
		self.filename = self.get('filename')
		self.writemethod = self.get('method', default="")
		self.keys = self.get('keys', default = [])
		self.configname = self.get('configuration', default = '')
		self.table = pd.DataFrame()
		self.meta = {}
		self.writerconfig = {}

	def _read_config(self):
		if self.configname:
			meta = yaml.safe_load(open(self.configname, "r").read())
			if "header" in meta:
				self.meta["header"] = meta["header"]
			if "parameters" in meta:
				self.meta["parameters"] = meta["parameters"]
			if "outfile" in meta:
				self.writerconfig = meta["outfile"]

		if "TableMetaData" in self.services:
			passedmeta = self.services["TableMetaData"]
			if "header" in passedmeta:
				if not "header" in self.meta:
					self.meta["header"] = passedmeta["header"]
				else:
					for key in passedmeta["header"]:
						self.meta["header"].setdefault(key, passedmeta["header"][key])
			if "parameters" in passedmeta:
				if not "parameters" in self.meta:
					self.meta["parameters"] = passedmeta["parameters"]
				else:
					for param in passedmeta["parameters"]:
						if param not in self.meta["parameters"]:
							self.meta["parameters"][param] = {}
						for key in passedmeta["parameters"][param]:
							self.meta["parameters"][param].setdefault(key, passedmeta["parameters"][key])

	def _get_dict_row(self, blob, initialize = False):
		if not self.keys:
			self.keys = blob.keys()
		inidict = {}
		for key in self.keys:
			if key in blob:
				if initialize:
					inidict.setdefault(key, [blob[key]])
				else:
					inidict.setdefault(key, blob[key])
		return inidict
		
	def _initialize_table(self, blob):
		self.table = pd.DataFrame(self._get_dict_row(blob, initialize=True), [0])

	def process(self, blob):
		if self.table.empty:
			self._initialize_table(blob)
		else:
			self.table = self.table.append(pd.DataFrame(self._get_dict_row(blob), [len(self.table)]), ignore_index = True)

	def finish(self):
		
		self._read_config()

		if not self.writemethod:
			if self.filename.split(".")[-1] in ("csv", "h5", "hdf5", "hdf", "fits"):
				self.writemethod = self.filename.split(".")[-1]
			else:
				self.writemethod = "csv"

		if self.writemethod == "csv":
			if not self.writerconfig:
				self.writerconfigs = {"path_or_buf":self.filename, "sep":"\t"}
			if "filename" in self.writerconfigs:
				self.writerconfigs["path_or_buf"]=self.writerconfigs["filename"]
				del self.writerconfigs["filename"]
			if not "path_or_buf" in self.writerconfig:
				self.writerconfigs["path_or_buf"] = self.filename
			self.table.to_csv(**self.writerconfigs)
			metafile = open(self.filename+"_meta.js", "w")
			metafile.write(json.dumps(self.meta))
			metafile.close()

		elif self.writemethod in ("h5", "hdf", "hdf5"):
			hfile = pd.HDFStore(self.filename)
			hfile.put("events", self.table)
			if self.meta:
				hfile.get_storer("events").attrs.metadata = self.meta
			hfile.close()

		elif self.writemethod == "fits":
			# TODO store in right format (type)
			columns = []
			for key in self.table.columns:
				col = np.array(self.table[key])
				columns.append(fits.Column(name=key, array = col, format=col.dtype))
			cols = fits.ColDefs(columns)
			fitsfile = fits.BinTableHDU.from_columns(cols)
			for key in self.meta["header"]:
				fitsfile.header.setdefault(key, self.meta["header"][key])
			fitsfile.writeto(self.filename)
			
			
