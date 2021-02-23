# -*- coding: utf-8 -*-
#Add basic tests here.

import os
for filename in os.listdir("../demos"):
    print('Running file: ' + filename)
    exec(open('../demos/'+filename).read())