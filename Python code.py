
# coding: utf-8

# # Programming Assignment 4B
# ## Decision Tree Classifier Without using API

# ### Data Used: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29/wdbc.data

# ##### Importing Required Libraries

# In[1]:


import findspark
findspark.init()
import numpy as np
import math
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#sc.stop()
conf = SparkConf().setMaster("local").setAppName("Dc")
sc = SparkContext(conf = conf)


# ## Reading Binned Data Prepared Using Excel VBA as Text file :: Please refer to the script screenshots provided in the zip
# 
# #### Please note that only the last 10 features are considered for model building : look into section(c) for further details.

# In[22]:


raw_data = sc.textFile('New_Data.csv')
print("Train data size is {}".format(raw_data.count()))
csv_data = raw_data.map(lambda x: x.split(","))


# In[23]:


print(csv_data.take(2))


# ## a) Program Code
# 
# 
# ### Splitting the cleaned data into 80:20 ratio of Training and Testing
# 

# In[24]:


(trainingData, testData) = csv_data.randomSplit([0.8, 0.2],seed=200)


# ### STAGE1 : Finding the entropy of the Dataset (E):
# #### Mapper :-  Key: Class_Column, Value : 1
# #### Reducer :- Key: Class_ , Value: Count

# In[5]:


mapper_output_1 = trainingData.map(lambda create_labeled_point:(create_labeled_point[10],1))
reducer_output_1 = mapper_output_1.reduceByKey(lambda a, b: a + b)


reducer_output_1.collect()


# ## Calculating the Entropy as part of controller module:

# In[6]:


E = -((166/467)*math.log((166/467),2))-((301/467)*math.log((301/467),2))
print(E)


# ## STAGE2 : Finding the Best Attribute at Level1:
# 
# #### Mapper  :- Key: Column_name, Column_Value,Class_Value  Value: 1
# #### Reducer:- Key: Column_name, Column_Value,Class_Value   Value: Count

# In[9]:


# Preparing Key
def defining_key(line):
    send=[]
    tail=['']
    for i in range(len(line)-1):
        s=str(i)+","+line[i]+","+line[10]
        send.append(s)
                
    return send    


mapper_output_2 = csv_data.map(lambda create_labeled_point:(create_labeled_point[0:11])).flatMap(defining_key)
reducer_output_2= mapper_output_2.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)


# ## Calculation at Controller:
# 
# ### Calculating the Entropy for each attribute as part of controller module: ( Used Pandas)

# In[10]:


df_reducer_output_2=pd.DataFrame(array(reducer_output_2.collect()))

df_reducer_output_2[['Feature','Value','Class']] = df_reducer_output_2[0].str.split(',', expand=True)
df_reducer_output_2['key']=df_reducer_output_2['Feature']+"_"+df_reducer_output_2['Value']


# In[14]:


Entr=[]
Entropy=[]
Entropy_level1=[]
for i in range(10):
    #print(i)
    Entropy=[]
    for j in range(7):
        Entr=[]
        a=str(i)+"_"+str(j)
        #print(a)
        if(len(df_reducer_output_2[(df_reducer_output_2['key']==a)])==2):
            #print("gf")
            #print(df[(df['hy']==a)])
            lb1=int(df_reducer_output_2[(df_reducer_output_2['key']==a) & (df_reducer_output_2['Class']=='1')][1])
            lb0=int(df_reducer_output_2[(df_reducer_output_2['key']==a) & (df_reducer_output_2['Class']=='0')][1])
            #print(lb0,lb1)
            x=lb0/(lb0+lb1)
            y=lb1/(lb0+lb1)
            e = -(x*math.log(x,2))-(y*math.log(y,2)) 
            Entr.append(a)
            Entr.append(e)
            Entr.append(lb1+lb0) 
            #print(Entr)
        if(len(df_reducer_output_2[(df_reducer_output_2['key']==a)])<=1) and (len(df_reducer_output_2[(df_reducer_output_2['key']==a)])>0):     
             lb=int(df_reducer_output_2[(df_reducer_output_2['key']==a)][1].tolist()[0])
             c=int(df_reducer_output_2[(df_reducer_output_2['key']==a)]['Class'].tolist()[0])
             e=0
             Entr.append(a)   
             Entr.append(e)
             Entr.append(c)
             #print(df[(df['hy']==a)]['hy'].tolist()[0],lb,c)   
        Entropy.append(Entr) 
        #print(Entropy)
    Entropy_level1.append(Entropy)
print(Entropy_level1)
            


# ## Finding Weighted entropy of each Attribute :: Printing Min weigted entropy for each attribute : 

# In[27]:


# w=0
avg=[]
for i in range(10):
        w=0
        for j in Entropy_level1[i]:
            w=w+(j[2]/569)*j[1]
        avg.append(w)

print("Minimum weighted Entropy for feature 8 among other weighted entropies as shown below :: " ,min(avg))
avg


# In[31]:


print("## Looking into the entropy of splits by attribute 8 :: ")
Entropy_level1[7]


# ## STAGE2 : Growing the Tree making Attribute 8 as the root node:
# 
# ### Task1 : Split the datasets as RDDs for all the values feature 8 would take. Here split0 indicates the data when feature 8 takes the value 0 and so on

# In[32]:


split0=csv_data.filter(lambda x:x[7]=="0")
split1=csv_data.filter(lambda x:x[7]=="1")
split2=csv_data.filter(lambda x:x[7]=="2")
split3=csv_data.filter(lambda x:x[7]=="3")
split4=csv_data.filter(lambda x:x[7]=="4")
split5=csv_data.filter(lambda x:x[7]=="5")
split6=csv_data.filter(lambda x:x[7]=="6")


# ### Task2 : Find the entropy at each split with every other attribute. Each function is used in stage2 but generalized for all the cases.

# In[35]:


def defining_key(line):
    send=[]
    for i in range(len(line)-1):
        s=str(i)+","+line[i]+","+line[10]
        send.append(s)              
    return send    

def Getting_Counts(dat):
    data1 = dat.map(lambda create_labeled_point:(create_labeled_point[0:11]))
    d=data1.flatMap(defining_key)
    c=d.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
    return c
def Get_Entropy(dat):
    a_ = array(Getting_Counts(dat).collect())
    df=pd.DataFrame(a_)
    df[['Feature','Value','Class']] = df[0].str.split(',', expand=True)
    df['key']=df['Feature']+"_"+df['Value']
    Entr=[]
    Entropy=[]
    Entropy1=[]
    for i in range(10):
        #print(i)
        Entropy=[]
        for j in range(7):
            Entr=[]
            a=str(i)+"_"+str(j)
            #print(a)
            if(len(df[(df['key']==a)])==2):
                lb1=int(df[(df['key']==a) & (df['Class']=='1')][1])
                lb0=int(df[(df['key']==a) & (df['Class']=='0')][1])
                #print(lb0,lb1)
                x=lb0/(lb0+lb1)
                y=lb1/(lb0+lb1)
                e = -(x*math.log(x,2))-(y*math.log(y,2))
                Entr.append(a)
                Entr.append(e)
                Entr.append(lb1+lb0)
                Entr.append(dat.count())
                if lb0>lb1:
                    pred=0
                    Entr.append(pred)
                else:
                    pred = 1
                    Entr.append(pred)  
            if(len(df[(df['key']==a)])<=1 and len(df[(df['key']==a)])>0): 
                 lb=int(df[(df['key']==a)][1].tolist()[0])
                 c=int(df[(df['key']==a)]['Class'].tolist()[0])
                 e=0
                 Entr.append(a)   
                 Entr.append(e)
                 Entr.append(lb)  
                 Entr.append(dat.count())
                 Entr.append(c)  
            Entropy.append(Entr) 
        Entropy1.append(Entropy)
    return Entropy1
    


# In[36]:


E0=Get_Entropy(split0)


print(E0)


# In[38]:


E1=Get_Entropy(split1)


# In[39]:


E2=Get_Entropy(split2)


# In[40]:


E3=Get_Entropy(split3)


# In[41]:


E4=Get_Entropy(split4)


# In[42]:


E5=Get_Entropy(split5)


# In[43]:


E6=Get_Entropy(split6)


# ## Task3 : Find Weighted Entropy at each split :: Returns the attribute with the minium entropy for each split

# In[45]:


# w=0

def Weighted_Entropy(Dat):
    avg=[]
    for i in range(10):
        #print(i)
        w=0
        list1= [x for x in Dat[i] if x]
        for j in list1:
            #print(j)
            w=w+(j[2]/j[3])*j[1]
            #print(w)  
        avg.append(w)
        #print(avg)
    return avg
#Entropy1[7]



# In[46]:


avg0=Weighted_Entropy(E0).index(min(Weighted_Entropy(E0)))
avg1=Weighted_Entropy(E1).index(min(Weighted_Entropy(E1)))
avg2=Weighted_Entropy(E2).index(min(Weighted_Entropy(E2)))
avg3=Weighted_Entropy(E3).index(min(Weighted_Entropy(E3)))
avg4=Weighted_Entropy(E4).index(min(Weighted_Entropy(E4)))
avg5=Weighted_Entropy(E5).index(min(Weighted_Entropy(E5)))
avg6=Weighted_Entropy(E6).index(min(Weighted_Entropy(E6)))


# In[51]:


print("Weighted entropy is stored in this way for each split:")
Weighted_Entropy(E0)


# In[53]:


print("Best Feautures found for each splits (array positions)",avg0,avg1,avg2,avg3,avg4,avg5,avg6)


# ## STAGE3 : Growing the Tree making Best attributes found in the earlier iterations:
# 
# ### For Split0 (ie., when feature_8 =0) Best attribute = 5(array position : 4)
# 
# ### For Split1 (ie., when feature_8 =1) Best attribute = 3(array position : 2)
# 
# ### For Split2 (ie., when feature_8 =2) Best attribute = 1(array position : 0)
# 
# ### For Split3 (ie., when feature_8 =3) Best attribute = 1(array position : 0)
# 
# ### For Split4 (ie., when feature_8 =4) Best attribute = 1(array position : 0)
# 
# ### For Split5 (ie., when feature_8 =5) Best attribute = 1(array position : 0)
# 
# ### For Split6 (ie., when feature_8 =6) Best attribute = 1(array position : 0)
# 
# ## Task1: Below are the RDDs for each of the branch further splists :: that is split00 - feature8 =0 and feature 5 =0 
# 
# ### Prepared from the RDDs of previous level

# In[ ]:


split00=split0.filter(lambda x:x[4]=="0")
split01=split0.filter(lambda x:x[4]=="1")
split02=split0.filter(lambda x:x[4]=="2")
split03=split0.filter(lambda x:x[4]=="3")
split04=split0.filter(lambda x:x[4]=="4")



split10=split1.filter(lambda x:x[2]=="0")
split11=split1.filter(lambda x:x[2]=="1")
split12=split1.filter(lambda x:x[2]=="2")



split20=split2.filter(lambda x:x[0]=="0")
split21=split2.filter(lambda x:x[0]=="1")
split22=split2.filter(lambda x:x[0]=="2")
split23=split2.filter(lambda x:x[0]=="3")


split30=split3.filter(lambda x:x[0]=="0")
split31=split3.filter(lambda x:x[0]=="1")
split32=split3.filter(lambda x:x[0]=="2")
split33=split3.filter(lambda x:x[0]=="3")
split34=split3.filter(lambda x:x[0]=="4")
split35=split3.filter(lambda x:x[0]=="5")
split36=split3.filter(lambda x:x[0]=="6")

split41=split4.filter(lambda x:x[0]=="1")
split42=split4.filter(lambda x:x[0]=="2")
split43=split4.filter(lambda x:x[0]=="3")
split44=split4.filter(lambda x:x[0]=="4")
split45=split4.filter(lambda x:x[0]=="5")

split51=split5.filter(lambda x:x[0]=="1")
split52=split5.filter(lambda x:x[0]=="2")
split53=split5.filter(lambda x:x[0]=="3")
split54=split5.filter(lambda x:x[0]=="4")
split55=split5.filter(lambda x:x[0]=="5")
split56=split5.filter(lambda x:x[0]=="6")

split62=split6.filter(lambda x:x[0]=="2")
split64=split6.filter(lambda x:x[0]=="4")
split65=split6.filter(lambda x:x[0]=="5")
split66=split6.filter(lambda x:x[0]=="6")


# ## Task2 : Finding entropy of each node in the level3

# In[56]:


E00=Get_Entropy(split00)
E01=Get_Entropy(split01)
E02=Get_Entropy(split02)
E03=Get_Entropy(split03)
E04=Get_Entropy(split04)


# In[58]:


E10=Get_Entropy(split10)
E11=Get_Entropy(split11)
E12=Get_Entropy(split12)


# In[60]:


E20=Get_Entropy(split20)
E21=Get_Entropy(split21)
E22=Get_Entropy(split22)
E23=Get_Entropy(split23)


# In[62]:


E30=Get_Entropy(split30)
E31=Get_Entropy(split31)
E32=Get_Entropy(split32)
E33=Get_Entropy(split33)
E34=Get_Entropy(split34)
E35=Get_Entropy(split35)
E36=Get_Entropy(split36)


# In[63]:


E41=Get_Entropy(split41)
E42=Get_Entropy(split42)
E43=Get_Entropy(split43)
E44=Get_Entropy(split44)
E45=Get_Entropy(split45)


# In[65]:


E51=Get_Entropy(split51)
E52=Get_Entropy(split52)
E53=Get_Entropy(split53)
E54=Get_Entropy(split54)
E55=Get_Entropy(split55)
E56=Get_Entropy(split56)


# In[67]:


E62=Get_Entropy(split62)
E64=Get_Entropy(split64)
E65=Get_Entropy(split65)
E66=Get_Entropy(split66)


# ### Task3 : Finding Weighted Entropy of each node in the level3

# In[68]:


avg00=Weighted_Entropy(E00).index(min(Weighted_Entropy(E00)))
avg01=Weighted_Entropy(E01).index(min(Weighted_Entropy(E01)))
avg02=Weighted_Entropy(E02).index(min(Weighted_Entropy(E02)))
avg03=Weighted_Entropy(E03).index(min(Weighted_Entropy(E03)))
avg04=Weighted_Entropy(E04).index(min(Weighted_Entropy(E04)))

avg10=Weighted_Entropy(E10).index(min(Weighted_Entropy(E10)))
avg11=Weighted_Entropy(E11).index(min(Weighted_Entropy(E11)))
avg12=Weighted_Entropy(E12).index(min(Weighted_Entropy(E12)))


avg20=Weighted_Entropy(E20).index(min(Weighted_Entropy(E20)))
avg21=Weighted_Entropy(E21).index(min(Weighted_Entropy(E21)))
avg22=Weighted_Entropy(E22).index(min(Weighted_Entropy(E22)))
avg23=Weighted_Entropy(E23).index(min(Weighted_Entropy(E23)))

avg30=Weighted_Entropy(E30).index(min(Weighted_Entropy(E30)))
avg31=Weighted_Entropy(E31).index(min(Weighted_Entropy(E31)))
avg32=Weighted_Entropy(E32).index(min(Weighted_Entropy(E32)))
avg33=Weighted_Entropy(E33).index(min(Weighted_Entropy(E33)))
avg34=Weighted_Entropy(E34).index(min(Weighted_Entropy(E34)))
avg35=Weighted_Entropy(E35).index(min(Weighted_Entropy(E35)))
avg36=Weighted_Entropy(E36).index(min(Weighted_Entropy(E36)))


avg41=Weighted_Entropy(E41).index(min(Weighted_Entropy(E41)))
avg42=Weighted_Entropy(E42).index(min(Weighted_Entropy(E42)))
avg43=Weighted_Entropy(E43).index(min(Weighted_Entropy(E43)))
avg44=Weighted_Entropy(E44).index(min(Weighted_Entropy(E44)))
avg45=Weighted_Entropy(E45).index(min(Weighted_Entropy(E45)))

avg51=Weighted_Entropy(E51).index(min(Weighted_Entropy(E51)))
avg52=Weighted_Entropy(E52).index(min(Weighted_Entropy(E52)))
avg53=Weighted_Entropy(E53).index(min(Weighted_Entropy(E53)))
avg54=Weighted_Entropy(E54).index(min(Weighted_Entropy(E54)))
avg55=Weighted_Entropy(E55).index(min(Weighted_Entropy(E55)))
avg56=Weighted_Entropy(E56).index(min(Weighted_Entropy(E56)))

avg62=Weighted_Entropy(E62).index(min(Weighted_Entropy(E62)))
avg64=Weighted_Entropy(E64).index(min(Weighted_Entropy(E64)))
avg65=Weighted_Entropy(E65).index(min(Weighted_Entropy(E65)))
avg66=Weighted_Entropy(E66).index(min(Weighted_Entropy(E66)))


# In[71]:


print("Best Attributes for split00-split04 ::",avg00,avg01,avg02,avg03,avg04)
print("Best Attributes for split10-split12 ::",avg10,avg11,avg12)
print("Best Attributes for split20-split23 ::",avg20,avg21,avg22,avg23)
print("Best Attributes for split30-split36 ::",avg30,avg31,avg32,avg33,avg34,avg35,avg36)
print("Best Attributes for split41-split45 ::",avg41,avg42,avg43,avg44,avg45)
print("Best Attributes for split51-split56 ::",avg51,avg52,avg53,avg54,avg55,avg56)
print("Best Attributes for split62-split66 ::",avg62,avg64,avg65,avg66)




# ## STAGE 4: Predicting the model on Test data(20%):
# ## The tree is written as the if else statements as below and the test data set is passed through to get predicted values

# In[28]:



Test_X=array(testData.collect())
#td=pd.DataFrame(b)
pred_=[]
for row in b:
    td=row
    if int(td[7])==0:
        pred==0
    if int(td[7])==1:
        if int(td[2])==2:
            pred=1
        else :
            pred=0
    if int(td[7])==2:
        if int(td[0])==0:
            pred=0
        else:
            pred = 1
    if int(td[7])==3:
        if int(td[0])==0 or int(td[0])==1:
            pred=0
        else:
            pred=1
    else:
        pred=1
    pred_.append(pred)        


# ##### Please note that the performance metrics are calculated in the section(f) at the bottom of the notebook

# ## b) The choice of parameters :
# * #### Impurity or Attribute Selection Method = "Entropy"
# * #### MaxDepth = 3 (Given)
# * #### Max Bins = 7

# ## c) Notes and Any assumptions made :
# 
# ### * Assumptions carried from Assignment 4A:: Reasoning why only last 10 features are selected for dataset.
# * The first feature ID is not contributing to the model hence ignored.
# * The features captured as Worst measure represent the data better than just measure and Standard error. Hence I have used only the columns from 22 till the end as my feature set.: radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst	compactness_worst,concavity_worst,concave points_worst,symmetry_worstfractal_dimension_worst.
# * I have validated the model on the other two sets (Mean and Standard Deviation) and found a better performance when I passed the worst_dimension features. You can look at the last section for the other runs.
# * The test errors are ::
# Mean : 0.097, SD : 0.141, Worst_ = 0.035( Smallest and hence the best gives the best features for model building)
# 
# ###  Assumptions for Assignment 4B and Comparision between 4A and 4B::
# * Considered the same bin size as that of the best bin size in Assignmnet4A. So each column is binned into 7 bins . This data preparation is done in excel vba. 
# * As per the ID3 algorithm, each split of the branch is grown, this is different from CART where only one attribute split is expanded each time. This results in the difference of results while an API is used.
# * In general binning does decrease the performance as there is information loss when the actual values are being replaced by the  bin values.
# * The leaf nodes obtained at the depth 3 are not the pure nodees in some of the splits, the prediction class is estimated based on the majority class at that particular node. To improve the accuary the tree has to further constructed.
# 

# ## d) Validation and Train/Test Startegy Used:
# 
#  ### Used the k-fold cross validation to evaluate the skill of the decision tree algorithm being learnt in general. I used the value k = 5 and the 4th fold seemed to be the best split giving out the accuracy upto 92.4%( Look into assumptions for my understanding of the decline of performance)
#  ### Used the criteria and split of that model in my program as parameters.
#  ### max_depth is fixed, min_samples_leaf  and criteria: entropy
#  
#  ## Below is the code for Cross Validation employed. Ran on the feature_worst Measures only(Last 10 feature set)

# In[8]:


File = 'C:\\Users\\yandr\\OneDrive\\Desktop\\BigData\\spark\\New_Data.csv'

df1 = pd.read_csv(File)

Train,Test = train_test_split(df1, test_size=0.3)
Data_X = Train.values[:,0:10]
Data_Y = Train.values[:,10]

X_Test=Test.values[:,0:10]
Y_Test=Test.values[:,10]

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[16]:


kf = KFold(n_splits=5,random_state=None, shuffle=True)
tree_fold = []
acc_tree=[]
prec_tree=[]
rec_tree=[]

def train_tree(X_train,X_test,Y_train,Y_test):
   tree = DecisionTreeClassifier(criterion = "entropy",max_depth=3,random_state = 200)
   tree.fit(X_train, Y_train)
   pred=tree.predict(X_test)
   tree_fold.append(tree)
   acc_tree.append(accuracy_score(Y_test,pred))
   prec_tree.append(precision_score(Y_test,pred,average= 'macro'))
   rec_tree.append(recall_score(Y_test,pred,average= 'macro')) 
   return

for train_index, test_index in kf.split(Data_X):
  #print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = Data_X[train_index], Data_X[test_index]
  Y_train, Y_test = Data_Y[train_index], Data_Y[test_index]

  train_tree(X_train,X_test,Y_train,Y_test)


# In[17]:


acc_tree,prec_tree,rec_tree
#Calculating the average Performances:
Avg_acc_tree= sum(acc_tree)/len(acc_tree)
Avg_prec_tree= sum(prec_tree)/len(prec_tree)
Avg_rec_tree= sum(rec_tree)/len(rec_tree)

print("Averge of the metrics:")

Avg_acc_tree,Avg_prec_tree,Avg_rec_tree
acc_tree,prec_tree,rec_tree


# In[18]:


pred_tree=tree_fold[3].predict(X_Test)
c_tree=confusion_matrix(Y_Test, pred_tree, labels=None, sample_weight=None)
acc_tree=accuracy_score(Y_Test,pred_tree)
prec_tree=precision_score(Y_Test,pred_tree,average= None)
rec_tree=recall_score(Y_Test,pred_tree,average= None)
print("Performance Using the best Fold:")

print("Accuracy =",acc_tree,"Precision =",prec_tree,"Recall =" ,rec_tree)

print("Confusion Matrix =")
print(c_tree)


# ## e) Decision tree Obtained: model is built in the pyspark. Look above for section a) for code
# 
# ####  * Please note that the features used are the last 10 features (Worst measure) from the dataset.

# In[44]:


#!pip install anytree
from anytree import Node, RenderTree

level1 = Node("Feature 7")
level2_0 = Node("Best Attr: Feature 4 (Data Based on: Feature7=0)", parent=level1)
level2_1 = Node("Best Attr: Feature 2 (Data Based on: Feature7=1)", parent=level1)
level2_2 = Node("Best Attr: Feature 0 (Data Based on: Feature7=2)", parent=level1)
level2_3 = Node("Best Attr: Feature 0 (Data Based on: Feature7=3)", parent=level1)
level2_4 = Node("Best Attr: Feature 0 (Data Based on: Feature7=4)", parent=level1)
level2_5 = Node("Best Attr: Feature 0 (Data Based on: Feature7=5)", parent=level1)
level2_6 = Node("Best Attr: Feature 0 (Data Based on: Feature7=6)", parent=level1)

level3_0_0 = Node("Best Attr: Feature 1 (Data Based on: Feature4=0)", parent=level2_0)
level3_0_1 = Node("Best Attr: Feature 0 (Data Based on: Feature4=1)", parent=level2_0)
level3_0_2 = Node("Best Attr: Feature 0 (Data Based on: Feature4=2)", parent=level2_0)
level3_0_3 = Node("Best Attr: Feature 0 (Data Based on: Feature4=3)", parent=level2_0)

level3_0_0_p = Node("Predict Label=0 ", parent=level3_0_0)
level3_0_1_p = Node("Predict Label=0 ", parent=level3_0_1)
level3_0_2_p = Node("Predict Label=0 ", parent=level3_0_2)
level3_0_3_p = Node("Predict Label=0 ", parent=level3_0_3)

level3_1_0 = Node("Best Attr: Feature 0 (Data Based on: Feature2=0)", parent=level2_1)
level3_1_1 = Node("Best Attr: Feature 1 (Data Based on: Feature2=1)", parent=level2_1)
level3_1_2 = Node("Best Attr: Feature 1 (Data Based on: Feature2=2)", parent=level2_1)

level3_1_0_p = Node("Predict Label=0 ", parent=level3_1_0)
level3_1_1_p = Node("Predict Label=0 ", parent=level3_1_1)
level3_1_2_p = Node("Predict Label=1 ", parent=level3_1_2)

level3_2_0 = Node("Best Attr: Feature 0 (Data Based on: Feature0=0)", parent=level2_2)
level3_2_1 = Node("Best Attr: Feature 1 (Data Based on: Feature0=1)", parent=level2_2)
level3_2_2 = Node("Best Attr: Feature 1 (Data Based on: Feature0=2)", parent=level2_2)
level3_2_3 = Node("Best Attr: Feature 0 (Data Based on: Feature0=3)", parent=level2_2)


level3_2_0_p = Node("Predict Label=0 ", parent=level3_2_0)
level3_2_1_p = Node("Predict Label=0 ", parent=level3_2_1)
level3_2_2_p = Node("Predict Label=0 ", parent=level3_2_2)
level3_2_3_p = Node("Predict Label=1 ", parent=level3_2_3)


level3_3_0 = Node("Best Attr: Feature 0 (Data Based on: Feature0=0)", parent=level2_3)
level3_3_1 = Node("Best Attr: Feature 1 (Data Based on: Feature0=1)", parent=level2_3)
level3_3_2 = Node("Best Attr: Feature 1 (Data Based on: Feature0=2)", parent=level2_3)
level3_3_3 = Node("Best Attr: Feature 0 (Data Based on: Feature0=3)", parent=level2_3)
level3_3_4 = Node("Best Attr: Feature 0 (Data Based on: Feature0=4)", parent=level2_3)
level3_3_5 = Node("Best Attr: Feature 0 (Data Based on: Feature0=5)", parent=level2_3)
level3_3_6 = Node("Best Attr: Feature 0 (Data Based on: Feature0=6)", parent=level2_3)


level3_3_0_p = Node("Predict Label=0 ", parent=level3_3_0)
level3_3_1_p = Node("Predict Label=0 ", parent=level3_3_1)
level3_3_2_p = Node("Predict Label=1 ", parent=level3_3_2)
level3_3_3_p = Node("Predict Label=1 ", parent=level3_3_3)
level3_3_4_p = Node("Predict Label=1 ", parent=level3_3_4)
level3_3_5_p = Node("Predict Label=1 ", parent=level3_3_5)
level3_3_6_p = Node("Predict Label=1 ", parent=level3_3_6)


level3_4_1 = Node("Best Attr: Feature 0 (Data Based on: Feature0=1)", parent=level2_4)
level3_4_2 = Node("Best Attr: Feature 0 (Data Based on: Feature0=2)", parent=level2_4)
level3_4_3 = Node("Best Attr: Feature 0 (Data Based on: Feature0=3)", parent=level2_4)
level3_4_4 = Node("Best Attr: Feature 0 (Data Based on: Feature0=4)", parent=level2_4)
level3_4_5 = Node("Best Attr: Feature 0 (Data Based on: Feature0=5)", parent=level2_4)


level3_4_1_p = Node("Predict Label=1 ", parent=level3_4_1)
level3_4_2_p = Node("Predict Label=1 ", parent=level3_4_2)
level3_4_3_p = Node("Predict Label=1 ", parent=level3_4_3)
level3_4_4_p = Node("Predict Label=1 ", parent=level3_4_4)
level3_4_5_p = Node("Predict Label=1 ", parent=level3_4_5)

level3_5_1 = Node("Best Attr: Feature 0 (Data Based on: Feature0=1)", parent=level2_5)
level3_5_2 = Node("Best Attr: Feature 0 (Data Based on: Feature0=2)", parent=level2_5)
level3_5_3 = Node("Best Attr: Feature 0 (Data Based on: Feature0=3)", parent=level2_5)
level3_5_4 = Node("Best Attr: Feature 0 (Data Based on: Feature0=4)", parent=level2_5)
level3_5_5 = Node("Best Attr: Feature 0 (Data Based on: Feature0=5)", parent=level2_5)
level3_5_6 = Node("Best Attr: Feature 0 (Data Based on: Feature0=6)", parent=level2_5)

level3_5_1_p = Node("Predict Label=1 ", parent=level3_5_1)
level3_5_2_p = Node("Predict Label=1 ", parent=level3_5_2)
level3_5_3_p = Node("Predict Label=1 ", parent=level3_5_3)
level3_5_4_p = Node("Predict Label=1 ", parent=level3_5_4)
level3_5_5_p = Node("Predict Label=1 ", parent=level3_5_5)
level3_5_6_p = Node("Predict Label=1 ", parent=level3_5_6)

level3_6_2 = Node("Best Attr: Feature 0 (Data Based on: Feature0=2)", parent=level2_6)
level3_6_4 = Node("Best Attr: Feature 0 (Data Based on: Feature0=4)", parent=level2_6)
level3_6_5 = Node("Best Attr: Feature 0 (Data Based on: Feature0=5)", parent=level2_6)
level3_6_6 = Node("Best Attr: Feature 0 (Data Based on: Feature0=6)", parent=level2_6)

level3_6_2_p = Node("Predict Label=1 ", parent=level3_6_2)
level3_6_4_p = Node("Predict Label=1 ", parent=level3_6_4)
level3_6_5_p = Node("Predict Label=1 ", parent=level3_6_5)
level3_6_6_p = Node("Predict Label=1 ", parent=level3_6_6)

for pre, fill, node in RenderTree(level1):
    print("%s%s" % (pre, node.name))


# ## f) Performance shown by the confusion matrix :

# In[47]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

a=Test_X[:,10]

actual=np.array(list(map(int, a)))


c_tree=confusion_matrix(actual1,actual, labels=None, sample_weight=None)
print("Confusion Matrix:")
print(c_tree)

acc_tree=accuracy_score(actual,actual1)
print("Accuracy:",acc_tree)

prec_tree=precision_score(actual,actual1,sample_weight=None)
print("Precision:",prec_tree)

rec_tree=recall_score(actual1,actual)
print("Recall:",rec_tree)


# ## References:
# 
# https://pypi.org/project/anytree/
