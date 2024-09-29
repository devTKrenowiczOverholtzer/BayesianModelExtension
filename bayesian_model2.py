import pandas as pd
import matplotlib.pyplot as plt 
# importing just the pyplot fo matplotlib
from sklearn.naive_bayes import GaussianNB
# guassian naive bayes classifier GaussianNB
from sklearn.model_selection import train_test_split
# test training split function
import seaborn as sns 
# seaborn to look at coorelation plot 

# dataset note: measurement data xmin xmax ymin ymax for each kind of steel defect (rows)
# dataset note: faults (scratch,stains,dirtiness,bumps,other)
# dataset note: column for each fault , 0 if the particular instance didnt display that fault and a 1 if it did 

#read in csv
df = pd.read_csv("faults.csv")
# print
print(df.head())

# Visualize correlation 
# print correlation corr()
##print(df.corr())

# Create heatmap of the correlation
##plot = sns.heatmap(df.corr())
# will cut off some of it becuase its not going to be enough 
##plt.show()
# clear
##plt.clf()

# see that there are a few features that are correlated with each other and the faults 
# want to be careful of eventually is these things which are really highly cross correlated because Bayesian asssumptions theres not a ton of cross correlation between your independent features or variables 
# create a classifer with all these features 
# extend model by using a couple different subsets of the features one of the ways I would suggest choosing those features is eliminating some of the ones that are cross correlated with others

# list out all the types of faults 
# k_scratch typo in dataset
# creating a list of faults 
faults = ["Pastry","Z_Scratch","K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
# Create new column : code each of these faults into its own integer in this column 
# going to have to create a different column that has an integer for each fault to put this into our passifier for sklearn
# data wrangling (many datasets)
df['fault'] = 0
# indexing our dataframe at this new column fault which we are creating and we are setting the whole column to 0 so every entry in the column will be equaled to zero 
# print columns 
print(df.columns)

# now have 35 columns 
# measurements from x min to sigmoidofareas 
# faults from pastry to other_faults 
# added column fault - print head see that this column should just have 0 there
print(df.head())

# dataset: sorted by fault type 
# default behavior of test train split function is to randomly grab a sample which is good 
# if we had just taken out the last 20% of the dataset it was sorted so the test set would have faults that werent seen in the training set which would be bad 

# Lets create a new categorical variable for each fault 
# This will be an integer
# This will go in the new fault column 
# make iterator in range 0 for length of faults list: iterate over faults list
for i in range(0, len(faults)):
	# get indexes of where all faults were that match a certain fault type 
	# index list
	# fault indexes are equal to the data frame and we are going to locate where the specific fault we are looking at is equal to 1 , going to get indexes of those and then convert it to a list
	true_fault_indexes = df.loc[df[faults[i]] == 1].index.tolist()
	# for each i start at 0 going to grab fault index 0 which first time it is going to be pastry, located in the dataframe everywhere where the pastry column is equal to 1, where pastry fault occured, grab the indexes of those in the dataset and convert this to a list.
	# Same with zscratch. will be i number 1 zscratch is fault number one index; find everywhere in our dataframe where our zscratch is equal to 1  cause that indicates a fault there and convert that to a list 
	# locate everywhere those indexs are and everywhere that the fault is true in the fault column going to create a categorical variable called i+1
	df.loc[true_fault_indexes, "fault"]=i+1
	# pastry i 0 index 0, everywhere there is a pastry fault in that fault column we are going to say is fault number 1 
	# just encoding it
print(df["fault"])
# when it goes to that iterator lets go ahead and print out fault column 
# first five faults are 1's thats good , thats pastry faults went in order 
# last five columns we see they are fault number 7 which are other faults 
# successfully assigned an integer encoding the type of fault into this fault column because this is the format that the classifier is going to expect it in  

# Create our dataset trainging and test x,y; features and outcomes 
# Create two sub datasets from our overall dataset 
# Our overall dataset encorporates both our features and our outcomes and we want to seperate just our features, sepearte just our outcomes 
# new list called drop features: fault column plus faults list: creating a list of things that wont be in the features dataframe 
# creating a list of things that wont be in the features data frame 
drop_features = ["fault"] + faults
# features new dataframe is going to be our dataframe where we drop our features we are going to drop and we are droppping it along the first axis
features = df.drop(drop_features, axis=1)
# outcomes equal to the dataframe fault column
outcomes = df["fault"]
# our df variable is not going to be changed
# but we are creating two sub dataframes that are going to be our inputs and our outputs to the model 
# the features data frame we dropped all the outputs 
# the outcome is resitricted to just this fault variable 

# columns of features
# x min to sigmoid area (no fault features)
print(features.columns) 
# head of outcomes 
# first five rows of fault we see have index fault variables that will correspond to the pastry faults 
print(outcomes.head())

training_features, test_features, training_outcomes, test_outcomes = train_test_split(features, outcomes, test_size=0.1)
# using sklearn split passing it features and outcomes to create these four sets and the set size is 10%
bayes_classifer = GaussianNB()
# create classifier which is guassianNB 
bayes_classifer.fit(training_features,training_outcomes)
#fit our training and testing features 
mean_accuracy = bayes_classifer.score(test_features, test_outcomes)
# print out score 
# when done score the test features and the test outcomes
#print mean_accuracy score - see how it did 
print(f"mean_accuracy: {mean_accuracy}")
# random subset of training and testing so this number can move around a little bit depending, rerun

# predict one instance with this function which is handy and also get the log probability of each outcome for each feature

