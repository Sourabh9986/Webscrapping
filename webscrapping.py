
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd, numpy as np


# In[19]:


book1= pd.read_csv(r'C:\Users\ghoshs20\Desktop\My Projects\PG\logistic reg\Book1.csv')
book1.head()


# In[20]:


varlist =  ['isscrappy']
def binary_map(x):
    return x.map({'Y': 1, "N": 0})

book1[varlist] = book1[varlist].apply(binary_map)


# In[21]:


book1.head()


# In[22]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(book1[['Webserver', 'CMS']], drop_first=True)

# Adding the results to the master dataframe
book1 = pd.concat([book1, dummy1], axis=1)

book1.head()


# In[24]:


book1=book1.drop(['CMS','Webserver'],1)


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


# Putting feature variable to X
X = book1.drop(['isscrappy','URL'], axis=1)

X.head()


# In[27]:


# Putting response variable to y
y = book1['isscrappy']

y.head()


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[29]:


# Checking scrappy rate

sum(book1['isscrappy'])/len(book1['isscrappy'])*100


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


plt.figure(figsize = (30,30))        # Size of the figure
sns.heatmap(book1.corr(),annot = True)
plt.show()


# In[36]:


import statsmodels.api as sm


# In[37]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[38]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[39]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 7)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)


# In[40]:


rfe.support_


# In[42]:


col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]


# In[43]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[44]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[45]:


y_train_pred_final = pd.DataFrame({'isscrappy':y_train.values, 'scrappy_Prob':y_train_pred})
y_train_pred_final['URL'] = y_train.index
y_train_pred_final.head()


# In[50]:


y_train_pred_final['predicted'] = y_train_pred_final.scrappy_Prob.map(lambda x: 1 if x > 0.45 else 0)

# Let's see the head
y_train_pred_final.head()


# In[47]:


from sklearn import metrics


# In[51]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.isscrappy, y_train_pred_final.predicted )
print(confusion)


# In[ ]:


# Predicted     not_scrappy    scrappy
# Actual
# not_scrappy        2153      1901
# scrappy            1193       1752  


# In[52]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.isscrappy, y_train_pred_final.predicted))


# In[53]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[54]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


col = col.drop('CMS_papaya CMS', 1)
col
col = col.drop('CMS_Fedora Commons', 1)
col


# In[63]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[64]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[65]:


y_train_pred_final['scrappy_Prob'] = y_train_pred


# In[108]:


# Creating new column 'predicted' with 1 if scrappy_Prob > 0.45 else 0
y_train_pred_final['predicted'] = y_train_pred_final.scrappy_Prob.map(lambda x: 1 if x > 0.45 else 0)
y_train_pred_final.head()


# In[109]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.isscrappy, y_train_pred_final.predicted))


# In[110]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[111]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.isscrappy, y_train_pred_final.predicted )
confusion


# In[ ]:


# Actual/Predicted     not_scrappy    scrappy
        # not_scrappy        2153      1901
        # scrappy            1193       1752 


# In[112]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.isscrappy, y_train_pred_final.predicted)


# In[113]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[114]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[115]:


# Let us calculate specificity
TN / float(TN+FP)


# In[116]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[117]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.isscrappy, y_train_pred_final.scrappy_Prob, drop_intermediate = False )


# In[118]:


draw_roc(y_train_pred_final.isscrappy, y_train_pred_final.scrappy_Prob)


# In[119]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.scrappy_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[120]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.isscrappy, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[121]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[122]:


y_train_pred_final['final_predicted'] = y_train_pred_final.scrappy_Prob.map( lambda x: 1 if x > 0.43 else 0)

y_train_pred_final.head()


# In[123]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.isscrappy, y_train_pred_final.final_predicted)


# In[124]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.isscrappy, y_train_pred_final.final_predicted )
confusion2


# In[125]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[126]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[127]:


# Let us calculate specificity
TN / float(TN+FP)


# In[128]:


#Make predictions on test data

X_test = X_test[col]
X_test.head()


# In[130]:


X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)


# In[131]:


y_test_pred[:10]


# In[132]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[133]:


# Let's see the head
y_pred_1.head()


# In[134]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[135]:


# Putting CustID to index
y_test_df['Url'] = y_test_df.index


# In[136]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[137]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[138]:


y_pred_final.head()


# In[139]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'scrappy_Prob'})


# In[140]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex_axis(['Url','isscrappy','scrappy_Prob'], axis=1)


# In[141]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[142]:


y_pred_final['final_predicted'] = y_pred_final.scrappy_Prob.map(lambda x: 1 if x > 0.43 else 0)


# In[143]:


y_pred_final.head()


# In[144]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.isscrappy, y_pred_final.final_predicted)


# In[145]:


confusion2 = metrics.confusion_matrix(y_pred_final.isscrappy, y_pred_final.final_predicted )
confusion2


# In[146]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[147]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[148]:


# Let us calculate specificity
TN / float(TN+FP)

