from activizer import app
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

from activizer.random_sampling import random_sampling as random_sampling
import numpy as np

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/download', methods=['GET', 'POST'])
def download():
    filename = 'result.csv'
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    return render_template("upload_model.html")

@app.route('/get_model', methods=['GET','POST'])
def get_model():
    file = request.files['file']
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    with open(os.path.join(UPLOAD_FOLDER, file.filename), 'rb') as handle:
        params = pickle.load(handle)
    
    accuracy = []
    accuracy.append(params["accuracy"])
    data = Data(params["counter"],params["X_pool"],params["y_pool"],params["learner"],params["committee"],accuracy
             ,params["X_test"],params["y_test"],params["classlist"],params["queries"],params["image_data"])

    return render_template("predict.html")


@app.route('/download_model', methods=['GET', 'POST'])
def download_model():
    filename = 'model.pickle'
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

@app.route('/display_image/',methods=['GET','POST'])
def display_image():
    parameters = request.args.to_dict()
    image = request.args.get('image')
    data = Data.getData()
    image_name_list = data.image_name_list
    found = False
    for i in image_name_list:
        if(i['image']==image):
            found = True
            image_array = i['image_array']
            label = i['label']
    if not found:
        label = parameters['label']
    im = Image.open(image)
    encoded_data = io.BytesIO()
    im.save(encoded_data, "JPEG")
    encoded_img_data = base64.b64encode(encoded_data.getvalue())
    return render_template("display_image.html",label=label,img_data=encoded_img_data.decode('utf-8'))



def generate_image():
    num_tiles = 20
    tile_size = 30
    arr = np.random.randint(0, 255, (num_tiles, num_tiles, 3))
    arr = arr.repeat(tile_size, axis=0).repeat(tile_size, axis=1)

    strIO = StringIO()
    buffer = BytesIO()
    imsave(buffer, arr, plugin='pil', format_str='png')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result",methods=['GET','POST'])
def result():
    data = Data.getData()
    learner = data.learner
    committee = data.committee
    classlist = data.classlist
    file = request.files['file']
    filename = secure_filename(file.filename)

    option = 0

    if file and allowed_file(file.filename):
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        if(filename.split(".")[1]=="rar"):
            option = 1
            patoolib.extract_archive(os.path.join(UPLOAD_FOLDER, filename), outdir=os.path.join(UPLOAD_FOLDER))
        elif(filename.split(".")[1]=="zip"):
            option = 1
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()

            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()

        else:
            filename = secure_filename(file.filename)
            im = Image.open(file)
            encoded_data = io.BytesIO()
            im.save(encoded_data, "JPEG")
            encoded_img_data = base64.b64encode(encoded_data.getvalue())
            image = im.resize((200,200), Image.ANTIALIAS)
            size = np.array(image).size
            x = numpy.array(image).reshape((1,size))
            if learner!=None:
                label = learner.predict(x)
            elif committee!=None:
                label = committee.predict(x)
            return render_template("display_image.html",label=classlist[label[0]]['name'],img_data=encoded_img_data.decode('utf-8'))
        if(option==1):
            list = []
            dict1 = {}
            for dirname, _, filenames in os.walk(os.path.join(UPLOAD_FOLDER,filename.split(".")[0])):
                for filename in filenames:
                    if('.jpg' in filename or 'jpeg' in filename or 'png' in filename):
                        image = Image.open(os.path.join(dirname, filename))
                        image = image.resize((200,200), Image.ANTIALIAS)
                        size = np.array(image).size
                        x = numpy.array(image).reshape((1,size))
                        try:
                            if learner!=None:
                                label = learner.predict(x)
                            elif committee!=None:
                                label = committee.predict(x)
                            list.append({"image":filename,"Label":classlist[label[0]]['name'],
                                         "image_name":os.path.join(dirname,filename)})
                            dict1[filename] = classlist[label[0]]['name']
                        except:
                            continue

        with open(os.path.join(UPLOAD_FOLDER,'result.csv'), 'w') as output:
            writer = csv.writer(output)
            for key, value in dict1.items():
                writer.writerow([key, value])
            return render_template("result.html",list = list)
    else:
        return render_template("result.html",name="Sorry")


@app.route("/")
def main():
    return render_template("index.html",data=[{'name':'Random Forest'}, {'name':'KNN'}, {'name':'Decision Tree'}],
                           query=[{'name':'Uncertainty Sampling'},{'name':'Entropy Sampling'},
                                    {'name':'Random Sampling'},
                                    {'name':'Query By Committee(Uncertainty Sampling)'},
                                    {'name':'Query By Committee(Vote Entropy Sampling)'},
                                    {'name':'Query By Committee(Max Disagreement Sampling)'},
                                    {'name':'Query By Committee(Consensus Entropy Sampling)'}
                                  ],
                           structure=[{'name':'Label Name given to Folder Containing Images','id':0},
                                  {'name':'Label Name given to Images','id':1}
                                  ])


@app.route('/train', methods=['POST'])
def helper():
    data = Data.getData()
    queries = data.queries
    X_test = data.X_test
    y_test = data.y_test
    X_pool = data.X_pool
    y_pool = data.y_pool
    counter = data.counter
    learner = data.learner
    committee = data.committee
    accuracy = data.accuracy
    classlist = data.classlist
    image_data = data.image_data
    image_name_list = data.image_name_list
    image_instance_name = data.image_instance_name
    image_instance = data.image_instance
    if(int(counter)==int(queries)):
        if(learner != None):
            query_idx, query_inst = learner.query(X_pool)
        elif(committee!=None):
            query_idx, query_inst = committee.query(X_pool)
        image_instance = query_inst
        image_instance_name = image_data[np.array(query_inst).tobytes()]
        try:
            arr = query_inst.reshape(200,200,3)
        except:
            arr = query_inst.reshape(200,200)
        rescaled = (255.0 / arr.max() * (arr - arr.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        new_size = (300, 300)
        im = im.resize(new_size)
        filename = secure_filename("image.png")
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        params = {}
        params["X_pool"] = X_pool
        params["y_pool"] = y_pool
        params["counter"] = int(counter)-1
        params["image_name_list"] = image_name_list
        params["image_instance_name"] = image_instance_name
        params["image_instance"] = image_instance
        if learner!=None:
            params["accuracy"] = learner.score(X_test,y_test)
        elif committee!=None:
            params["accuracy"] = committee.score(X_test, y_test)
        data.setdata(params)
        im = Image.open(image_instance_name)
        encoded_data = io.BytesIO()
        im.save(encoded_data, "JPEG")
        encoded_img_data = base64.b64encode(encoded_data.getvalue())
        return render_template("after.html",classlist=classlist,UPLOAD_FOLDER=os.path.join(UPLOAD_FOLDER,"image.png"),img_data=encoded_img_data.decode('utf-8'))
    elif(int(counter)>0):
        if(learner != None):
            query_idx, query_inst = learner.query(X_pool)
        elif(committee!=None):
            query_idx, query_inst = committee.query(X_pool)
        try:
            arr = query_inst.reshape(200,200,3)

        except:
            arr = query_inst.reshape(200,200)
        rescaled = (255.0 / arr.max() * (arr - arr.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        new_size = (300, 300)
        im = im.resize(new_size)
        filename = secure_filename("image.png")


        y_new = np.array([int(request.form.get('label_select'))],dtype=int)
        image_label = int(request.form.get('label_select'))
        image_name_list.append({"image":image_instance_name,"label":classlist[image_label]['name'],"image_array":image_instance})

        image_instance_name = image_data[np.array(query_inst).tobytes()]
        image_instance = query_inst
        im = Image.open(image_instance_name)

        encoded_data = io.BytesIO()
        im.save(encoded_data, "JPEG")
        encoded_img_data = base64.b64encode(encoded_data.getvalue())
        if(learner!=None):
            learner.teach(query_inst.reshape(1, -1), y_new)
        elif(committee!=None):
            committee.teach(query_inst.reshape(1, -1), y_new)
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        params = {}
        params["X_pool"] = X_pool
        params["y_pool"] = y_pool
        params["counter"] = int(counter)-1
        params["image_name_list"] = image_name_list
        params["image_instance_name"] = image_instance_name
        params["image_instance"] = image_instance
        if learner!=None:
            params["accuracy"] = learner.score(X_test,y_test)

        elif committee!=None:
            params["accuracy"] = committee.score(X_test, y_test)
        data.setdata(params)
        accuracy_string = ""
        count = 0
        iterations = ""
        for i in data.accuracy:
            n = float(i)
            n*=100
            accuracy_string +=str(n)
            accuracy_string +=","
            iterations+=str(count)
            iterations+=","
            count+=1
        accuracy_string = accuracy_string[:-1]
        iterations = iterations[:-1]
        return render_template("after.html",data = accuracy_string,iteration = iterations,classlist=classlist,
                               UPLOAD_FOLDER=os.path.join(UPLOAD_FOLDER,"image.png"),img_data=encoded_img_data.decode('utf-8'))
    else:
        y_new = np.array([int(request.form.get('label_select'))],dtype=int)
        image_label = int(request.form.get('label_select'))
        image_name_list.append({"image":image_instance_name,"label":classlist[image_label]['name'],"image_array":image_instance})
        params = {}
        params["X_pool"] = X_pool
        params["y_pool"] = y_pool
        params["counter"] = int(counter)-1
        params["X_test"] = X_test
        params["y_test"] = y_test
        params["queries"] = queries
        params["classlist"] = classlist
        params["image_data"] = image_data
        params["image_name_list"] = image_name_list
        params["image_instance_name"] = image_instance_name
        params["image_instance"] = image_instance
        if learner!=None:
            params["accuracy"] = learner.score(X_test,y_test)
        elif committee!=None:
            params["accuracy"] = committee.score(X_test, y_test)
        data.setdata(params)

        accuracy_string = ""
        iterations = ""
        count = 0
        for i in data.accuracy:
            n = float(i)
            n *= 100
            accuracy_string += str(n)
            accuracy_string += ","
            iterations += str(count)
            iterations += ","
            count += 1
        accuracy_string = accuracy_string[:-1]
        iterations = iterations[:-1]
        prediction_model = {}
        prediction_model["classlist"] = classlist
        if learner!=None:
            params["learner"] = learner
            params["committee"] = learner

        else:
            params["committee"] = committee
            params["learner"] = committee
        with open(os.path.join(UPLOAD_FOLDER,'model.pickle'), 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return render_template("final.html",accuracy = float(data.accuracy[-1])*100,data = accuracy_string,iteration = iterations,
                               list = image_name_list)


@app.route('/next',methods=['GET','POST'])
def query():
    strategy = None
    classifier = None

    file = request.files['file']
    test = request.files['test_file']
    filename = secure_filename(file.filename)
    test_filename = secure_filename(test.filename)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
        zip_ref.extractall(UPLOAD_FOLDER)
        zip_ref.close()

    
    if test and allowed_file(test.filename):
        filename = secure_filename(test.filename)
        test.save(os.path.join(UPLOAD_FOLDER, filename))
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
        zip_ref.extractall(UPLOAD_FOLDER)
        zip_ref.close()
    

    st = request.form.get('strategy_select')
    cl = request.form.get('classifier_select')
    option = int(request.form.get('structure_select'))
    if(str(cl)=='Random Forest'):
        classifier = RandomForestClassifier()
    elif(str(cl)=='KNN'):
        classifier = KNeighborsClassifier()
    else:
        classifier = DecisionTreeClassifier()

    n_queries = request.form['queries']

    classlist =[]
    classes = {}
    data = {}
    data['image'] = []
    data['label'] = []
    data['image_name'] = []
    image_data = {}
    filename = secure_filename(file.filename)
    if option == 0:
        for root,dirs,filename in os.walk(os.path.join(UPLOAD_FOLDER,filename.split(".")[0])):
            for name in filename:
                if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
                    image_file_name = os.path.join(root,name)
                    image = Image.open(image_file_name)
                    image = image.resize((200,200), Image.ANTIALIAS)
                    size = np.array(image).size

                    if(len(classes)==0):
                        data['image'] = np.array(numpy.array(image)).reshape((1,size))
                        image_data[(numpy.array(image).reshape((1,size))).tobytes()] = image_file_name
                    else:
                        try:
                            x = numpy.array(image).reshape((1,size))
                            image_data[(numpy.array(image).reshape((1,size))).tobytes()] = image_file_name
                            data['image'] = np.append(data['image'],x,axis=0)
                        except:
                            continue
                    if root.split("\\")[-1] not in classes.keys():
                        classlist.append({'name':root.split('\\')[-1],'number':len(classes)})
                        classes[root.split('\\')[-1]] = len(classes)

                    data['label'].append(classes[root.split('\\')[-1]])
                    data['image_name'].append(image_file_name)
    else:
        for imfile in os.listdir(os.path.join(UPLOAD_FOLDER,filename.split(".")[0])):

            if imfile.endswith(".jpg") or imfile.endswith(".jpeg") or imfile.endswith(".png"):
                image_file_name = os.path.join(os.path.join(UPLOAD_FOLDER,filename.split(".")[0]), imfile)
                image = Image.open(os.path.join(os.path.join(UPLOAD_FOLDER,filename.split(".")[0]), imfile))
                image = image.resize((200,200), Image.ANTIALIAS)
                size = np.array(image).size

                if(len(classes)==0):
                    data['image'] = np.array(numpy.array(image)).reshape((1,size))
                    image_data[(numpy.array(image).reshape((1,size))).tobytes()] = image_file_name
                else:
                    try:
                        x = numpy.array(image).reshape((1,size))
                        image_data[(numpy.array(image).reshape((1,size))).tobytes()] = image_file_name
                        data['image'] = np.append(data['image'],x,axis=0)
                    except:
                        continue
                if(("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0]))) not in classes.keys()):
                    classlist.append({'name':("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0]))),'number':len(classes)})
                    classes[("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0])))] = len(classes)
                data['label'].append(classes[("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0])))])
                data['image_name'].append(imfile)
            else:
                continue

    test_classlist =[]
    test_classes = {}
    test_data = {}
    test_data['image'] = []     
    test_data['label'] = []
    test_data['image_name'] = []
    if option == 0:
        for dirname, _, filenames in os.walk(os.path.join(UPLOAD_FOLDER,test_filename.split(".")[0])):
            for filename in filenames:
                if('.jpg' in filename or 'jpeg' in filename or 'png' in filename):
                    image = Image.open(os.path.join(dirname, filename))
                    image = image.resize((200,200), Image.ANTIALIAS)
                    size = np.array(image).size
                    if(len(test_classes)==0):
                        test_data['image'] = np.array(numpy.array(image)).reshape((1,size))
                    else:
                        try:
                            x = numpy.array(image).reshape((1,size))
                            test_data['image'] = np.append(test_data['image'],x,axis=0)
                        except:
                            continue
                    if(dirname.split('\\')[-1] not in test_classes.keys()):
                        test_classlist.append({'name':dirname.split('\\')[-1],'number':len(test_classes)})
                        test_classes[dirname.split('\\')[-1]] = len(test_classes)
    
                    test_data['label'].append(test_classes[dirname.split('\\')[-1]])
                    test_data['image_name'].append(filename)

    else:
        for imfile in os.listdir(os.path.join(UPLOAD_FOLDER,test_filename.split(".")[0])):
            if imfile.endswith(".jpg") or imfile.endswith(".jpeg") or imfile.endswith("png"):
                image = Image.open(os.path.join(os.path.join(UPLOAD_FOLDER,test_filename.split(".")[0]), imfile))
                image = image.resize((200,200), Image.ANTIALIAS)
                size = np.array(image).size

                if(len(test_classes)==0):
                    test_data['image'] = np.array(numpy.array(image)).reshape((1,size))
                else:
                    try:
                        x = numpy.array(image).reshape((1,size))
                        test_data['image'] = np.append(test_data['image'],x,axis=0)
                    except:
                        continue
                if(("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0]))) not in test_classes.keys()):
                    test_classlist.append({'name':("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0]))),'number':len(test_classes)})
                    test_classes[("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0])))] = len(test_classes)
                test_data['label'].append(test_classes[("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0])))])
                test_data['image_name'].append(imfile)

            else:
                continue
    X_train = data['image']
    y_train = data['label']
    X_test = test_data['image']
    y_test = test_data['label']
    n_initial = 100

    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial=[]
    y_initial = []
    for i in range(n_initial):

        v = np.array(X_train[initial_idx[i]]).reshape((1,size))

        y_initial.append(y_train[i])
        if(i==0):
            X_initial = np.array(X_train[initial_idx[i]]).reshape((1,size))


        else:
            X_initial = np.append(X_initial,v,axis=0)

    X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

    params = {}
    params["X_test"] = X_test
    params["y_test"] = y_test
    params["counter"] = n_queries
    params["X_pool"] = X_pool
    params["y_pool"] = y_pool
    if(str(st)=='Uncertainty Sampling'):

        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=uncertainty_sampling,
            X_training=X_initial, y_training=y_initial
        )

        params["learner"] = learner
        accuracy_scores = learner.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,learner,None,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()
    elif(str(st)=='Entropy Sampling'):

        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=entropy_sampling,
            X_training=X_initial, y_training=y_initial
        )

        params["learner"] = learner
        accuracy_scores = learner.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,learner,None,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()
    elif(str(st)=='Random Sampling'):
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=random_sampling,
            X_training=X_train, y_training=y_train
        )
        accuracy_scores = learner.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,learner,None,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()
    elif(str(st)=='Query By Committee(Vote Entropy Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=vote_entropy_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()

    elif(str(st)=='Query By Committee(Uncertainty Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=uncertainty_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()

    elif(str(st)=='Query By Committee(Max Disagreement Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
                query_strategy=max_disagreement_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()


    elif(str(st)=='Query By Committee(Consensus Entropy Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=consensus_entropy_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries,image_data)
        return helper()
