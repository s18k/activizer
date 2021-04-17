
class Data:
    __data = None
    @staticmethod
    def getData():
        if Data.__data == None:
            Data()
        return Data.__data
    def __init__(self,counter,X_pool,y_pool,learner,committee,accuracy,X_test,y_test,classlist,queries,image_data):

        self.counter = counter
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.learner = learner
        self.committee = committee
        self.accuracy = list(accuracy)
        self.X_test = X_test
        self.y_test = y_test
        self.classlist = classlist
        self.queries = queries
        self.image_data = image_data
        self.image_name_list = []
        self.image_instance_name = ""
        self.image_instance = None
        Data.__data = self

    def setdata(self,params):
        self.counter = params["counter"]
        self.X_pool = params["X_pool"]
        self.y_pool = params["y_pool"]
        self.image_name_list = params["image_name_list"]
        self.image_instance_name = params["image_instance_name"]
        self.image_instance = params["image_instance"]
        list = self.accuracy
        list.append(params["accuracy"])
        self.accuracy = list
        Data.__data = self


    def givedata(self):
        params={}
        params["counter"] = self.counter
        params["X_pool"] = self.X_pool
        params["y_pool"] = self.y_pool
        params["learner"] = self.learner
        params["committee"] = self.committee
        params["accuracy"] = self.accuracy
        return params

