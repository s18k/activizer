
import base64
import pickle
from io import StringIO

import io
from flask import Flask, send_file
import csv
import numpy

from skimage.io import imsave

from io import BytesIO
from PIL import Image
import os
from activizer.data import Data

import re
from pathlib import Path


app = Flask(__name__, static_url_path="", static_folder="static ")
app.secret_key = "super secret key"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLD = 'C:/Users/ASUS/PycharmProjects/Project_Data'


UPLOAD_FOLDER = str(Path.home() / "Downloads")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import zipfile
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['zip','rar','png','jpeg','jpg'])
from modAL.uncertainty import uncertainty_sampling,entropy_sampling
from modAL.disagreement import vote_entropy_sampling,max_disagreement_sampling,max_std_sampling,consensus_entropy_sampling

from modAL.models import ActiveLearner, Committee

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)

import activizer.views
