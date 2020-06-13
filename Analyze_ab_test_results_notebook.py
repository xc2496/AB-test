#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[4]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[5]:


df['converted'].sum()/290584


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


treat_old = df.query("group == 'treatment' and landing_page == 'old_page'").shape[0]
control_new = df.query("group == 'control' and landing_page == 'new_page'").shape[0]

treat_old + control_new


# f. Do any of the rows have missing values?

# In[7]:


df.info()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df2 = df.query("group == 'control' and landing_page == 'old_page'")
df2 = df2.append(df.query("group == 'treatment' and landing_page == 'new_page'"))


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


df2[df2['user_id'].duplicated()]


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2[df2['user_id'] == 773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


df2 = df2.drop(1899)


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


control_prob = df2.query("group == 'control'")['converted'].mean()
control_prob


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treatment_prob = df2.query("group == 'treatment'")['converted'].mean()
treatment_prob


# d. What is the probability that an individual received the new page?

# In[17]:


df2.query('landing_page == "new_page"').shape[0]/df2.shape[0]


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# I don't think there is a sufficient evidence to conclude that the new treatment page leads to more conversions. First, the new page (treatment) led to a lower conversion rate than the old page (control). In addition the difference between two groups in concersion rate are small.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# <p>$$H_0: p_{old} = p_{new} $$</p>
# 
# 

# <p>$$H_1: p_{old} \lt p_{new}$$</p>

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[18]:


p_null = df2['converted'].mean()
p_null


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[19]:


p_null


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[20]:


n_new = df2.query("landing_page == 'new_page'").shape[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[21]:


n_old = df2.query("landing_page == 'old_page'").shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[22]:


new_page_converted = np.random.binomial(1, p_null, n_new)


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[23]:


old_page_converted = np.random.binomial(1, p_null, n_old)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[24]:


new_page_converted.mean() - old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[25]:


p_diffs = []
new_converted_simulation = np.random.binomial(n_new, p_null, 10000)/n_new
old_converted_simulation = np.random.binomial(n_old, p_null, 10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[26]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[27]:


diff = treatment_prob - control_prob

low_prob = (p_diffs < diff).mean()
high_prob = (p_diffs.mean() + (p_diffs.mean() - diff) < p_diffs).mean()

plt.hist(p_diffs);
plt.axvline(obs_diff, color='red');
plt.axvline(p_diffs.mean() + (p_diffs.mean() - diff), color='red');

p_val = low_prob + high_prob
print(p_val)


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# This value is called the p-value.  The p-value is a number between 0 and 1 and interpreted in the following way: A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis. It appeared that the p-value is above 0.05 (0.1907), which means we fail to reject the null hypothesis ( $P{new} = P_{old}$). And conclude that there is not a significant different between the new and old pages.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[ ]:


import statsmodels.api as sm


convert_old = df2.query("landing_page == 'old_page'")['converted'].sum()
convert_new = df2.query("landing_page == 'new_page'")['converted'].sum()
n_old = df2.query("landing_page=='old_page'").user_id.nunique()
n_new = df2.query("landing_page=='new_page'").user_id.nunique()


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[ ]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new])
print(z_score, p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

#  A z-score (also called a standard score) gives you an idea of how far from the mean a data point is. But more technically it’s a measure of how many standard deviations below or above the population mean a raw score is. For here, it represents the difference between conversion rates and the null hypothesis is 1.31 standard deviations above the mean. The critical z score is 1.96, so we would need to reject the null hypothesis. Additionally, the p-value here is 0.1899, which is greater than 0.05. So we reject the null hyposthesis as well. We found the p value as 0.1907, which is similar to 0.1899. These there is an agreement in part j and k.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# We will perform the logistic regression since there is a True or False option for the results in the variable.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[ ]:


df2['intercept'] = 1
df2[['a_page', 'ab_page']] = pd.get_dummies(df2['group'])
df2 = df2.drop('a_page', axis=1)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[ ]:


log_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = log_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[ ]:


results.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# The p-value of ab_page is 0.1899 here, which is very similar to the previous observation, but a little bit higher. This model is trying to predict whether a user will convert depending on their page. The null hypothesis is that when ab_page = 1, converted = 0; the alternative hypothesis is that when ab_page = 1, converted is more likely to be 1.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# So far, it does not show that the treatment or control page has much influence on whether a user converts. Thus, it is probably a good idea to see whether other things that might influence whether or not an individual converts. So when we select the factors, it is important to see the correlations.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[ ]:


#read the countries.csv into the project and adding countries into as a factor
countries = pd.read_csv('countries.csv')
countries.head()
df2 = df2.set_index('user_id').join(countries.set_index('user_id'))


# In[ ]:


#Set dummy variables
df2[['CA', 'UK', 'US']] = pd.get_dummies(df2['country'])


# In[ ]:


logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page', 'CA', 'UK']])
results = logit_mod.fit()
results.summary2()


# Based on the p-values above 0.05 ($P_{CA} = 0.1295$, $P_{UK}=0.4573$).So there is not a significant differences of conversions between countires. 

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[ ]:


df2['CA_page'] = df2['CA']*df2['ab_page']
df2['UK_page'] = df2['UK']*df2['ab_page']
df2['US_page'] = df2['US']*df2['ab_page']
logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'CA_page', 'UK_page']])
results = logit_mod.fit()
results.summary2()


# Based on the above, the p-value of the interaction between CA and ab_page does present as statistically significant(p = 0.0458; p < 0.05).
# 
# 

# In[ ]:


# get results param
np.exp(results.params)


# In[ ]:


1 / np.exp(results.params)


# According tothe CA_page coefficient,holding all other variables constant, a canadian user who gets the new page would be about 1.078 times more likely to convert than other countires. While this holds a small amount of statistical significance, it does not hold much practical significance. Since it is also the only value of statistical significance. The ab_page and country were not significant respectively. In summary, we fail reject the null hypothesis based on any of our A/B testing. So there is no evidence to support to switch to the new page.

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

