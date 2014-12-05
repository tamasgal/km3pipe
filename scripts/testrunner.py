#!/usr/bin/env python

import sys
#sys.path.append('..')
import unittest


loader = unittest.TestLoader()
tests = loader.discover('..')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
