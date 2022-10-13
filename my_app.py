
from flask import Flask, url_for ,render_template


from datetime import datetime
import os
import urllib
import numpy as np
import cv2
import pyodbc
import skimage.measure
import face_recognition
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_local
import pytesseract 


app = Flask(__name__)
#app.debug = True

# class PrefixMiddleware(object):
# #class for URL sorting 
#     def __init__(self, app, prefix=''):
#         self.app = app
#         self.prefix = prefix

#     def __call__(self, environ, start_response):
#         #in this line I'm doing a replace of the word flaskredirect which is my app name in IIS to ensure proper URL redirect
#         if environ['PATH_INFO'].lower().replace('/flaskapp','').startswith(self.prefix):
#             environ['PATH_INFO'] = environ['PATH_INFO'].lower().replace('/flaskapp','')[len(self.prefix):]
#             environ['SCRIPT_NAME'] = self.prefix
#             return self.app(environ, start_response)
#         else:
#             start_response('404', [('Content-Type', 'text/plain')])            
#             return ["This url does not belong to the app.".encode()]


# app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='')

@app.route('/test')
def test():
    return "123"


@app.route('/bar')
def bar():
    try:
        uid = '6'
        conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;''uid=sa;pwd=sa@123')
        #conn = pyodbc.connect(r'Driver={SQL Server};Server=DELLSERVER\SQLEXPRESS;Database=ReferralSoftware;Trusted_Connection=yes;')

        cursor = conn.cursor()
        #userid = int(uid)
        cursor.execute('SELECT * FROM UserVerification where Uid='+uid)
        row = cursor.fetchone()

     

        
        
        return row[2]
    except Exception as e:
        return str(e)




@app.route('/sigrecog')
def sigrecog():
    try:
        id = "Veriid122"
        conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=SignatureVerification;''uid=sa;pwd=sa@123')
        
        savingpath="C:/inetpub/wwwroot/Flask_IIS/VerificationImages/sig/"
        #conn = pyodbc.connect('Driver={SQL Server};' 'Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;' 'uid=sa;pwd=sa@123')
      #  conn = pyodbc.connect(r'Driver={SQL Server};Server=DELLSERVER\SQLEXPRESS;Database=ReferralSoftware;Trusted_Connection=yes;')

        cursor = conn.cursor()
        #userid = int(uid)
        cursor.execute('SELECT * FROM SignatureVerification where Uid='+id)
        row = cursor.fetchone()

        impath1 = savingpath+uid+"sg1.jpg"
        impath2 = savingpath+uid+"sg2.jpg"
        resource = urllib.request.urlopen(row[2])
        output = open(impath1,"wb")
        output.write(resource.read())
        output.close()

        resource = urllib.request.urlopen(row[3])
        output = open(impath2,"wb")
        output.write(resource.read())
        output.close()
        #imgt = cv2.imread(impath2, 0)
        total = 0
        matched = 0
            
        query_img = cv2.imread(impath1) 
        train_img = cv2.imread(impath2) 
        query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
        query_img_bw = cv2.threshold(query_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
        train_img_bw = cv2.threshold(train_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
        orb = cv2.ORB_create() 
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

        matcher = cv2.BFMatcher() 
        matches = matcher.match(queryDescriptors,trainDescriptors) 

        total = max(len(queryDescriptors),len(trainDescriptors))
        matched = len(matches)
        
        if (matched/total)*100>70:
            cursor.execute("UPDATE SignatureVerification SET Verification = 1 WHERE Uid="+uid)
            conn.commit()
            print("m")
            return "Mached"
        else:
            cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
            conn.commit()
            print("nm")
            return "Not matched"
        
    except Exception as e:
        print(e)
        cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
        conn.commit()
        return "Not matched" 


def removeallfiles(d):
    filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
    for f in filesToRemove:
        os.remove(f) 




def dist(a,b,mode='L2'):
    if mode == 'L2':
        return np.linalg.norm(a-b,axis = 1)
    
def surf_feat(path):
    surf_features = []
    surf_kps = []
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000,extended = 1)
    dirc = os.listdir(path)
    cnt = 0
    img_store = []
    for img in dirc:
        imgname = path+img
        #print(imgname)
        if img[-4:] == '.png' or img[-4:]=='.PNG' or img[-4:] == '.jpg' or img[-4:] == '.jpeg':
            tmp = cv2.imread(imgname,  0)
            #print(tmp)
            img_store.append(tmp)
            kp, des = surf.detectAndCompute(tmp,None)
            surf_features.append(des)
            surf_kps.append(kp)
            cnt+=1
    surf_features = np.asarray(surf_features)
    #print(surf_features)
    surf_kps = np.array(surf_kps)
    stable_descs = []
    stable_kps = []
    unstable_kps = []
    
    for i in range(cnt):
        idxs = list(range(cnt))
        idxs.remove(i)
        #print(surf_features[idxs])
        features_loo = np.concatenate(surf_features[idxs])
        tot_dist = 0;
        dists = []
        for descs in surf_features[i]:
            min_dist = np.min(dist(features_loo,descs))
            tot_dist+=min_dist
            dists.append(min_dist)
        dist_avg = tot_dist/cnt
        stable_descs.extend(list(surf_features[i][np.where(dists<=dist_avg)]))
        stable_idx = np.where(dists<=dist_avg)[0]
        unstable_idx = np.where(dists>dist_avg)[0]
        stable_kps.append([surf_kps[i][j] for j in stable_idx])
        unstable_kps.append([surf_kps[i][j] for j in unstable_idx])
    
    return np.array(stable_descs)




def classify(query_path, stable_descs, threshold = 0.11, batch_size = 'Full', num_plots = 5):
    queries = os.listdir(query_path)
    if batch_size != 'Full':
        queries = queries[:batch_size]
    percentage_match = []
    for i, query_name in enumerate(queries):
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000,extended = 1)
        print(query_path+query_name)
        img = cv2.imread(query_path+query_name,  cv2.IMREAD_GRAYSCALE)
       # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #print(img)
        kp, des = surf.detectAndCompute(img, None)
        tot_des = des.shape[0]
        matches = 0
        stable_kp = []
        unstable_kp = []
        itr=0
        pts = [[int(p.pt[0]), int(p.pt[1])] for p in kp]
        #print(des)
        for desc in des:
            min_dist = np.min(dist(stable_descs,desc))
            if min_dist <= threshold:
                matches+=1
                stable_kp.append(pts[itr])
            else:
                unstable_kp.append(pts[itr])
            itr+=1
        stable_kp = np.asarray(stable_kp)
        unstable_kp = np.asarray(unstable_kp)
        print(matches,tot_des)
        
        percentage_match.append(matches/tot_des)
    return np.array(percentage_match)


@app.route('/sigmatching/<id>')
def signatureverification(id):
    try:
        uid = id
        conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=SignatureVerification;''uid=sa;pwd=sa@123')
        
        savingpath="C:/inetpub/wwwroot/Flask_IIS/VerificationImages/sig/"
        #conn = pyodbc.connect('Driver={SQL Server};' 'Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;' 'uid=sa;pwd=sa@123')
      #  conn = pyodbc.connect(r'Driver={SQL Server};Server=DELLSERVER\SQLEXPRESS;Database=ReferralSoftware;Trusted_Connection=yes;')

        cursor = conn.cursor()
        #userid = int(uid)
        cursor.execute('SELECT * FROM SignatureVerification where Uid='+id)
        row = cursor.fetchone()

        impath1 = savingpath+uid+"sg1.jpg"
        impath2 = savingpath+uid+"sg2.jpg"
        resource = urllib.request.urlopen(row[2])
        output = open(impath1,"wb")
        output.write(resource.read())
        output.close()

        resource = urllib.request.urlopen(row[3])
        output = open(impath2,"wb")
        output.write(resource.read())
        output.close()
        #imgt = cv2.imread(impath2, 0)
        total = 0
        matched = 0
            
        query_img = cv2.imread(impath1) 
        train_img = cv2.imread(impath2) 
        query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
        query_img_bw = cv2.threshold(query_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
        train_img_bw = cv2.threshold(train_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
        orb = cv2.ORB_create() 
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

        matcher = cv2.BFMatcher() 
        matches = matcher.match(queryDescriptors,trainDescriptors) 

        total = max(len(queryDescriptors),len(trainDescriptors))
        matched = len(matches)
        
        if (matched/total)*100>70:
            cursor.execute("UPDATE SignatureVerification SET Verification = 1 WHERE Uid="+uid)
            conn.commit()
            print("m")
            return "Mached"
        else:
            cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
            conn.commit()
            print("nm")
            return "Not matched"
        
    except Exception as e:
        print(e)
        cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
        conn.commit()
        return "Not matched"        
    


@app.route('/outputfin/<data>')
def output(data):
 
    try:
        uid,stype = data.split(",")
        if stype == "jobzone":
            conn = pyodbc.connect('Driver={SQL Server};''Server=blocdrivedb.database.windows.net;''Database=Jobzone;''uid=Blocdrive;pwd=Srichid@123')
        elif stype == "matrimony":
            conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=MatrimonyUSM;''uid=sa;pwd=sa@123')
        else:
            conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;''uid=sa;pwd=sa@123')
        #conn = pyodbc.connect(r'Driver={SQL Server};Server=DELLSERVER\SQLEXPRESS;Database=ReferralSoftware;Trusted_Connection=yes;')

        cursor = conn.cursor()
        #userid = int(uid)
        cursor.execute('SELECT * FROM UserVerification where Uid='+uid)
        row = cursor.fetchone()

     
        # make a list of all the available images

        # load your image

       
        resource = urllib.request.urlopen(row[2])
        output = open("C:/inetpub/wwwroot/Flask_IIS/VerificationImages/img/"+uid+"1.jpg","wb")
        output.write(resource.read())
        output.close()
        image_to_be_matched = face_recognition.load_image_file('C:/inetpub/wwwroot/Flask_IIS/VerificationImages/img/'+uid+'1.jpg')

        # encoded the loaded image into a feature vector
        image_to_be_matched_encoded = face_recognition.face_encodings(
            image_to_be_matched)[0]

        #print(image_to_be_matched_encoded)

        # iterate over each image
        resource = urllib.request.urlopen(row[3])
        output = open("C:/inetpub/wwwroot/Flask_IIS/VerificationImages/img/"+uid+"2.jpg","wb")
        output.write(resource.read())
        output.close()
        current_image = face_recognition.load_image_file( 'C:/inetpub/wwwroot/Flask_IIS/VerificationImages/img/'+uid+'2.jpg')
       
            # encode the loaded image into a feature vector
        current_image_encoded = face_recognition.face_encodings(current_image)[0]
       
            # match your image with the image and check if it matches
        result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
            # check if it was a match]
            
    
        if result[0] == True:
            cursor.execute("UPDATE UserVerification SET Verification = 1 WHERE Uid="+uid)
            conn.commit()
            return ("Matched" )
        else:
            cursor.execute("UPDATE UserVerification SET Verification = 0 WHERE Uid="+uid)
            conn.commit()
            return ("Not matched" )
    except Exception as e:
        print(e)
        cursor.execute("UPDATE UserVerification SET Verification = 0 WHERE Uid="+uid)
        conn.commit()
        return str(e)


@app.route('/sigmatchfin/<data>')
def sigmatch(data):
    try:
        uid,stype = data.split(",")
        if stype == "jobzone":
            conn = pyodbc.connect('Driver={SQL Server};''Server=blocdrivedb.database.windows.net;''Database=Jobzone;''uid=Blocdrive;pwd=Srichid@123')
        elif stype == "matrimony":
            conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=MatrimonyUSM;''uid=sa;pwd=sa@123')
        else:
            conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;''uid=sa;pwd=sa@123')
        
        savingpath="C:/inetpub/wwwroot/Flask_IIS/VerificationImages/sig/"
        #conn = pyodbc.connect('Driver={SQL Server};' 'Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;' 'uid=sa;pwd=sa@123')
      #  conn = pyodbc.connect(r'Driver={SQL Server};Server=DELLSERVER\SQLEXPRESS;Database=ReferralSoftware;Trusted_Connection=yes;')

        cursor = conn.cursor()
        #userid = int(uid)
        cursor.execute('SELECT * FROM SignatureVerification where Uid='+uid)
        row = cursor.fetchone()

        impath1 = savingpath+uid+"sg1.jpg"
        impath2 = savingpath+uid+"sg2.jpg"
        resource = urllib.request.urlopen(row[2])
        output = open(impath1,"wb")
        output.write(resource.read())
        output.close()

        resource = urllib.request.urlopen(row[3])
        output = open(impath2,"wb")
        output.write(resource.read())
        output.close()
        #imgt = cv2.imread(impath2, 0)
        total = 0
        matched = 0
        crop_img = cropImage(impath2)
        print(len(crop_img))
        if len(crop_img)>0:
            cv2.imwrite(savingpath+"cpre"+ uid +".png",crop_img)
            gray_image = cv2.imread(savingpath+"cpre"+ uid +".png",0)
            ret,thresh1 = cv2.threshold(gray_image,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(savingpath+"cpres"+ uid +".png",thresh1)
            
            query_img = cv2.imread(impath1) 
            train_img = cv2.imread(savingpath+"cpres"+ uid +".png") 
            query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
            query_img_bw = cv2.threshold(query_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
            train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
            train_img_bw = cv2.threshold(train_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
            orb = cv2.ORB_create() 
            queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
            trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

            matcher = cv2.BFMatcher() 
            matches = matcher.match(queryDescriptors,trainDescriptors) 

            total = max(len(queryDescriptors),len(trainDescriptors))
            matched = len(matches)
            
            if (matched/total)*100>70:
                cursor.execute("UPDATE SignatureVerification SET Verification = 1 WHERE Uid="+uid)
                conn.commit()
                print("m")
                return "1"
            else:
                cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
                conn.commit()
                print("nm")
                return "0"
                
        else:
            cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
            conn.commit()
            print("null")
            return "0"
        
    except Exception as e:
        print(e)
        cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
        conn.commit()
        return str(e)        
    
    
    
def cropImage(path):
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    rgb = cv2.imread(path)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    im2 = rgb.copy()
    #threshold the image
    _, bw = cv2.threshold(small, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # get horizontal mask of large size since text are horizontal components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # find all the contours
    contours, hierarchy=cv2.findContours(connected.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]


    height,width,_ =im2.shape 

    signatureBytes = []
    #Segment the text lines
    tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w] 
        cv2.imwrite("C:/inetpub/wwwroot/Flask_IIS/VerificationImages/test/"+str(idx)+".png",cropped)
        try:
            text = pytesseract.image_to_string(cropped, config = tessdata_dir_config)
            #print(text)

            # Apply OCR on the cropped image
            if 'signature' in str(text).lower() or 'signature'==str(text).lower():
                #print(idx,text,x,y,h,w)

                sp = (x/width)*100


                if sp>40:


                    ax = int((width*10)/100)
                    ay= int((height*16)/100)

                    w+=int((width*2)/100)
                    ax1 = x - ax
                else:

                    ax = int((width*15)/100)
                    ay= int((height*15)/100)

                    if x-ax >0:
                        w+=int((width*12)/100)
                        ax1 = x - ax
                    else:
                        w+=int((width*12)/100)
                        ax1 = 0
                ay1 = y - ay
                cropped1 = rgb[ay1:y -20 , ax1:x + w]
                signatureBytes = cropped1
                break
           
        except Exception as e:
            continue
    return signatureBytes

@app.route('/sigmatchfss/<data>')
def sigmatchfss(data):
    try:
        uid,stype = data.split(",")
        if stype == "jobzone":
            conn = pyodbc.connect('Driver={SQL Server};''Server=blocdrivedb.database.windows.net;''Database=Jobzone;''uid=Blocdrive;pwd=Srichid@123')
        elif stype == "matrimony":
            conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=MatrimonyUSM;''uid=sa;pwd=sa@123')
        else:
            conn = pyodbc.connect('Driver={SQL Server};''Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;''uid=sa;pwd=sa@123')
        
        savingpath="C:/inetpub/wwwroot/Flask_IIS/VerificationImages/sig/"
        #conn = pyodbc.connect('Driver={SQL Server};' 'Server=DELLSERVER\SQLEXPRESS;''Database=ReferralSoftware;' 'uid=sa;pwd=sa@123')
      #  conn = pyodbc.connect(r'Driver={SQL Server};Server=DELLSERVER\SQLEXPRESS;Database=ReferralSoftware;Trusted_Connection=yes;')

        cursor = conn.cursor()
        #userid = int(uid)
        cursor.execute('SELECT * FROM SignatureVerification where Uid='+uid)
        row = cursor.fetchone()

        impath1 = savingpath+uid+"sg1.jpg"
        impath2 = savingpath+uid+"sg2.jpg"
        resource = urllib.request.urlopen(row[2])
        output = open(impath1,"wb")
        output.write(resource.read())
        output.close()

        resource = urllib.request.urlopen(row[3])
        output = open(impath2,"wb")
        output.write(resource.read())
        output.close()
        #imgt = cv2.imread(impath2, 0)
        total = 0
        matched = 0

        gray_image = cv2.imread(impath1,0)
        ret,thresh1 = cv2.threshold(gray_image,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(savingpath+"cpres"+ uid +".png",thresh1)
        
        query_img = cv2.imread(impath1) 
        train_img = cv2.imread(savingpath+"cpres"+ uid +".png") 
        query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
        query_img_bw = cv2.threshold(query_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
        train_img_bw = cv2.threshold(train_img_bw, 127, 255, cv2.THRESH_BINARY)[1]
        orb = cv2.ORB_create() 
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

        matcher = cv2.BFMatcher() 
        matches = matcher.match(queryDescriptors,trainDescriptors) 

        total = max(len(queryDescriptors),len(trainDescriptors))
        matched = len(matches)
        
        if (matched/total)*100>70:
            cursor.execute("UPDATE SignatureVerification SET Verification = 1 WHERE Uid="+uid)
            conn.commit()
            print("m")
            return "1"
        else:
            cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
            conn.commit()
            print("nm")
            return "0"

        
    except Exception as e:
        print(e)
        cursor.execute("UPDATE SignatureVerification SET Verification = 0 WHERE Uid="+uid)
        conn.commit()
        return "0"    


def thr(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) Threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
   
    return threshed

def clrim(img):
    #img = cv2.imread(path,0)
    edges = cv2.Canny(img,100,200)
   
    return edges
           



def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB):
    s = ssim(imageA, imageB, multichannel= True)
    m = mse(imageA, imageB)
   
    #print('mse = {} and ssim = {}'.format(m, s))
    if int(s*100)>85:
        return "Matched"
    else:
        return "not matched"


@app.route('/sigverfin', methods=["GET","POST"])
def sigverfin():
    # vid = "abcd"
   
    data = request.get_json()
    vid = data['vid']
    org = data['link1']
    comp = data['link2']

    try:
        savingpath = "D:/home/site/vishal/"  # change this path

        impath1 = savingpath + vid + "1.jpg"
        impath2 = savingpath + vid + "2.jpg"
        impath3 = savingpath + vid + "11.jpg"
        impath4 = savingpath + vid + "12.jpg"

        resource = urllib.request.urlopen(org)
        output = open(impath1, "wb")
        output.write(resource.read())
        output.close()

        resource = urllib.request.urlopen(comp)
        output = open(impath2, "wb")
        output.write(resource.read())
        output.close()
       
        original = clrim1(thr1(impath1))
        contrast = clrim1(thr1(impath2))

        s1 = original.shape
        s2 = contrast.shape

        w = 0
        h = 0

        if s1[1] >= s2[1]:
            w = s2[1]
        else:
            w = s1[1]
        if s1[0] >= s2[0]:
            h = s2[0]
        else:
            h = s1[0]

        dim = (w, h)
        img1 = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(contrast, dim, interpolation=cv2.INTER_AREA)
       
       
       
        val = (compare_images1(img1, img2))
       
        return val

    except Exception as e:
        return "0"


def thr1(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) Threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return threshed


def clrim1(img):
    # img = cv2.imread(path,0)
    edges = cv2.Canny(img, 100, 200)

    return edges


def mse1(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images1(imageA, imageB):
    s = ssim(imageA, imageB, multichannel=True)
    m = mse1(imageA, imageB)

    # print('mse = {} and ssim = {}'.format(m, s))
    #if int(s * 100) > 85:
    #    return str(s*100)
    #else:
    #    return "not matched"
    return str(s*100)

def getsamples(img):
    x, y, z = img.shape
    samples = np.empty([x * y, z])
    index = 0
    for i in range(x):
        for j in range(y):
            samples[index] = img[i, j]
            index += 1
    return samples


def EMSegmentation(img, no_of_clusters=2):
    output = img.copy()
    colors = np.array([[0, 11, 111], [22, 22, 22]])
    samples = getsamples(img)
    em = cv2.ml.EM_create()
    em.setClustersNumber(no_of_clusters)
    em.trainEM(samples)
    means = em.getMeans()
    covs = em.getCovs()  
    x, y, z = img.shape
    distance = [0] * no_of_clusters
    for i in range(x):
        for j in range(y):
            for k in range(no_of_clusters):
                diff = img[i, j] - means[k]
                distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
            output[i][j] = colors[distance.index(max(distance))]
    return output




	
if __name__ == '__main__':
    app.run()
