'''
# FIS FINAL PROJECT
# TOPIC : RECOMMENDATION SYSTEM
# TEAM MEMBER:
# NIRAJ DEDHIA (njd4185)
# SIDDHARTH BIDWALKAR (ssb9012)
# SANDEEP CHAUDHARY (ssc4763)
#
'''
import csv
import numpy as np
import math;

'''
# predictPlacesForUser() :
# Here we are using the Hybrid recommendation method : Hybrid recommendation is developed by combining the
# concepts of Content based filtering and Collaborative filtering.
#
# Algorithmic steps:
# 1) Content based Filtering for predicting the taste of user:
#   - Reading individual user's rating for each place one by one.
#   - Calculating Feature vector for all places.
#   - Calculating Parameter Vector for all the users using linear regression.
#   - This method will help us to predict the user's taste.
#
# 2) Collaborative Filtering to calculate nature of place:
#   - Calculated Similar users who share the same rating patterns with the active user.
#   - Built place-place matrix which determines how similar the one place is from other.
#   - Based on the Current user taste filtered out the places.
#
# 3) Hybrid Filtering:
#   - Combined both the methods to implement hybrid filtering
#   - First Content based is ran to infer the taste of user.
#   - Then we found similarities between places and similar users using collaborative.
#   - Found out all the places visited by other user but not visited by active user.
#   - Predicted ratings for the places with respect to user taste.
#   - Filtered out the place list based on the predited ratings.
#   - Returned all the places with the best possible match.
#
# @param : user id
#
# @return : List of predicted places of user's taste.
#
'''
def predictPlacesForUser(user):

    userFile = 'userprofile.csv'; # File contain user profile
    placeFile = 'geoplaces2.csv'; # File contains places and its features
    ratingFile = 'rating_final.csv'; # Actually rating file

    # === This factor use for filter out the places
    predi = 1.2; # Display all the places whose prediction is greater then predi
    # ===

    userList = []; # All the users
    placeList = []; # All the places
    placeNameList = []; # Name of all the places
    X = []; # User's food and Service for visited places
    Y = []; # USer's total rating for visited places

#==================================Reading User and Place profile from .CSV=============================================
    # Reading User Profile file and storing it in user List
    with open(userFile) as csvfile:
         reader = csv.DictReader(csvfile);
         for row in reader:
             userList.append(row['userID']); # Feeding user lists
             X.append([]);
             Y.append([]);

    # Reading Places file and storing it in placeList List
    with open(placeFile) as csvfile:
         reader = csv.DictReader(csvfile);
         for row in reader:
             placeList.append(row['placeID']); # Feeding place list
             name = "\n NAME : "+row['name'];
             address = "\n ADDRESS : " + row['address'];
             url = "\n URL :"+ row['url'];

             PlaceDis = str(name+address+url);
             placeNameList.append(PlaceDis); # Feeding place description

#=======================================================================================================================

    y = []; # y(i,j) is rating by user j on movie i
    nFeature = 2; # number of features

#====================================Initializing y and place features==================================================
    placeFeatures = []; # All the features for places
    placeFeatureCounters = []; # Total number of features

    # Initializing y array
    for j in range(0,len(placeList)):
        list = [];
        for i in range(0,len(userList)):
            list.append(-999);
        y.append(list);
        lF = [];
        lC = [];
        k = 0;
        while(k<nFeature):
            lF.append(0);
            lC.append(0);
            k += 1;
        placeFeatures.append(lF);
        placeFeatureCounters.append(lC);

#=======================================================================================================================

    placesListsVistedByEachUser = {}; # List of places visited by each users

#==============================================Feeding ratings==========================================================

    # Reading User rating file and masking all the ratings on y
    # Filling user j rating for movie i in y(i,j)
    with open(ratingFile) as csvfile:
        reader = csv.DictReader(csvfile);

        for row in reader:

            j = (placeList.index(row['placeID']));
            i = (userList.index(row['userID']));

            placeListVisitByUseri = [];
            if (placesListsVistedByEachUser.__contains__(userList[i])): # If user has visited any other place then just append
                placeListVisitByUseri = placesListsVistedByEachUser[userList[i]];

            if(int(row['rating']) > 0): # If user has given ratings > 0
                placeListVisitByUseri.append(placeList[j]);
            placesListsVistedByEachUser[userList[i]] = placeListVisitByUseri;

            foodInt = int(row['food_rating']);
            serviceInt = int(row['service_rating']);
            ratingInt = int(row['rating']);

            # First Feature : Food
            y[j][i] = ratingInt ;
            placeFeatures[j][0] +=  foodInt;
            placeFeatureCounters[j][0] +=  1;

            # Second Feature : Service
            placeFeatures[j][1] +=  serviceInt;
            placeFeatureCounters[j][1] +=  1;

            # Feeding X and Y lists
            listX = X[i];
            l = [1,foodInt,serviceInt]
            listX.append(l);
            X[i] = listX;

            listY = Y[i];
            listY.append(ratingInt);
            Y[i] = listY;

#=======================================================================================================================


    x = []; # Feature vector for each movie
    for i in range(0,len(placeList)):
        lis = [];
        x.append(lis);

#========================================Calculating Feature vectors for movie==========================================


    for i in range(0,len(placeList)): # For each movie calculate average of food and service features
        x[i].append((float)(1))
        if(placeFeatures[i][0] != 0):
            x[i].append( (float) ( placeFeatures[i][0] / placeFeatureCounters[i][0] ) );
        else:
            x[i].append(0);

        if(placeFeatures[i][1] != 0):
            x[i].append( (float) ( placeFeatures[i][1] / placeFeatureCounters[i][1] )  );
        else:
            x[i].append(0);

#=======================================================================================================================

    theta = []; # parameter factor for user i

#=================================Calculating parameter fator for each user=============================================

    for i in range(0,len(userList)):
        # Applying Linear Regression to calculate the value for parameter factor for each user
        # Y = Transpose(theta) * (X)
        th = np.dot(np.linalg.pinv((np.dot(np.transpose(X[i]),X[i]))),np.dot(np.transpose(X[i]),Y[i]))
        theta.append(th);

#=======================================================================================================================

#==============================================Users Compare============================================================

    similarUsers = {}; # Storing user having similar tastes
    for i in range (0,len(userList)):
        l = [];
        for j in range (i+1,len(userList)):
            # Calculating Distance by Euclidian distance formula
            d = math.sqrt(math.pow((theta[i][0]-theta[j][0]),2) + math.pow((theta[i][1]-theta[j][1]),2) + math.pow((theta[i][2]-theta[j][2]),2) );
            if(d<=0.4 and d>=-0.2):
                l.append(userList[j]);
        similarUsers[userList[i]] = l;

#=======================================================================================================================

#===============================================Users and places =======================================================

    listOFSimilarUser = similarUsers[user];
    totalPlaces = []; # total places visited by similar users
    li = placesListsVistedByEachUser[user];
    for i in listOFSimilarUser:
        for j in placesListsVistedByEachUser[i]:
            if(j not in totalPlaces and j not in li):
                totalPlaces.append(j);

    bestPlaces = []
    # Finding the best places for user out of received total places by using user parameter and feature vectors
    for place in totalPlaces:
        prediction = np.dot( theta[userList.index(user)] , x[placeList.index(place)] );
        if(prediction >= predi):
            bestPlaces.append(place);

    for i in range(0,len(bestPlaces)): # Reading the place description
        bestPlaces[i] = placeNameList[placeList.index(bestPlaces[i])];

    return (bestPlaces);
#=======================================================================================================================



'''
# main() : It is the main method which will get called initially
# It will take the user id and which dispalys the list of similar places
# suggested for the respected user.
'''
def main():

#===============================================Test Cases==============================================================

    #user = 'U1082'
    user = input("Enter the User to whome we wish to recommend places: \n");
    print("USER : ", user);
    list = predictPlacesForUser(user);
    if (len(list) == 0):
        print("No match found");
    else:
        print("Total : ",len(list)," matches found \n");
        print("We recommend '",user,"' will like following places");
        for l in list:
            print(l);
#=======================================================================================================================





if __name__ == '__main__':
    main()