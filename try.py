def model_selection(df_input):
    results = {}
    l = list(df_input.columns)
    l.remove('label')
    X_train, X_test, y_train, y_test = train_test_split(df_input[l], df_input['label'],
                                                    test_size=0.33, random_state=42)
    
    ## BNB
    clf_bnb = BernoulliNB().fit(X_train, y_train)
    ypred=clf_bnb.predict(X_test)
    acc_BNB = accuracy_score(y_test,ypred)
    print("Accuracy for BNB model with/without tuning: ", acc_BNB)
    results[clf_bnb] = acc_BNB
    
    ## Random Forest
    clf_rf = RandomForestClassifier(random_state=42)
    param_grid = [{ 
        'n_estimators': np.arange(10,110,10),
        'max_depth' : [4,5,6,7,8]
    }]
    CV_rf = GridSearchCV(estimator=clf_rf, param_grid=param_grid, cv= 5)
    CV_rf.fit(X_train, y_train)
    y_pred = CV_rf.predict(X_test)
    acc_RF = accuracy_score(y_test,y_pred)
    print("Accuracy for Random Forest after CV: ", acc_RF)
    results[CV_rf] = acc_RF
    
    ## KNN
    clf_knn = KNeighborsClassifier()
    param_grid = [{ 
        'n_neighbors': np.arange(8,20)
    }]
    CV_knn= GridSearchCV(estimator=clf_knn, param_grid=param_grid, cv= 5)
    CV_knn.fit(X_train, y_train)
    y_pred = CV_knn.predict(X_test)
    acc_KNN = accuracy_score(y_test,y_pred)
    print("Accuracy for KNN after CV data: ",acc_KNN)
    results[CV_knn] = acc_KNN
    
    ## Voting Classifier
    eclf1 = VotingClassifier(estimators=[('nb', clf_bnb), ('rf', CV_rf), ('knn', CV_knn)], voting='hard')
    eclf1 = eclf1.fit(X_train, y_train)
    ypred = eclf1.predict(X_test)
    acc_VC = accuracy_score(y_test,ypred)
    print("Accuracy for voting classifier: ",acc_VC)
    results[eclf1] = acc_VC
    
    best_estimator = max(results, key=lambda k: results[k])
    return best_estimator
    
