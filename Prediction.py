def prediction(reviews, itemType):
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer
    # import contractions
    import joblib
    import re
    import pymongo
    import datetime
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    item = itemType
    
    fitDataset = pd.read_csv('Datasets/amazonreviews.tsv', delimiter = '\t', quoting = 3)
    fitCorpus = []
    for i in fitDataset['review']:
        review = re.sub('[^a-zA-Z]', ' ', i )
        review = review.lower()
        # review = contractions.fix(review)
        review = word_tokenize(review)
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer() 
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) 
                  for word in review if not word in set(all_stopwords)]
        review = [lemmatizer.lemmatize(word) 
                  for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        fitCorpus.append(review)
    
    dataset = pd.DataFrame(reviews)
    corpus = []
    for i in dataset['Reviews']:
        review = re.sub('[^a-zA-Z]', ' ', i)
        review = review.lower()
        # review = contractions.fix(review)
        review = review.split()
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word)
                  for word in review if not word in set(all_stopwords)]
        review = [lemmatizer.lemmatize(
            word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    filename = 'Model.sav'
    loaded_model = joblib.load(open(filename, 'rb'))

    Phones = ['iphone 7 plus', 'iphone 6', 'huawei y9']
    Laptops = ['hp i7', 'hp i5', 'dell i5']

    Feature1 = []
    Feature2 = []
    Feature3 = []

    if(item in Phones):
        device = Phones
        feature1 = 'display'
        feature2 = 'batteri'
        feature3 = 'speaker'
        featureOne = 'battery'
        featureTwo = 'desplay'
        featureThree = 'speakers'
    else:
        device = Laptops
        feature1 = 'display'
        feature2 = 'batteri'
        feature3 = 'charger'
        featureOne = 'battery'
        featureTwo = 'desplay'
        featureThree = 'charger'

    for comment in corpus:
        x = re.search(feature1, comment)
        y = re.search(feature2, comment)
        z = re.search(feature3, comment)
        if x:
            Feature1.append(comment)
        if y:
            Feature2.append(comment)
        if z:
            Feature3.append(comment)
            
    cv = CountVectorizer(max_features = 1500)
    cv.fit_transform(fitCorpus).toarray()
    
    def p(x):
        pos_list = []
        pos_count = 0
        for i in x:
            if (i == 'pos'):
                pos_list.append(i)
                pos_count += 1
        return pos_count

    def n(y):
        neg_list = []
        neg_count = 0
        for i in y:
            if (i == 'neg'):
                neg_list.append(i)
                neg_count += 1
        return neg_count

    def count(Feature):
        if(Feature):
            Feature = cv.transform(Feature).toarray()
            PredictFeature = loaded_model.predict(Feature)
            pos_count_Feature = p(PredictFeature)
            neg_count_Feature = n(PredictFeature)
            total_count = pos_count_Feature + neg_count_Feature
            pos_ratio = int((pos_count_Feature / total_count) * 100)
            neg_ratio = int((neg_count_Feature / total_count) * 100)
            return pos_ratio, neg_ratio
        else:
            pos_ratio = 0
            neg_ratio = 0
            return pos_ratio, neg_ratio
        
    count = count(Feature1)
    pos_ratio1 = count[0]
    neg_ratio1 = count[1]
    
    count = count(Feature2)
    pos_ratio2 = count[0]
    neg_ratio2 = count[1]
    
    count = count(Feature3)
    pos_ratio3 = count[0]
    neg_ratio3 = count[1]
    
    count = count(corpus)
    pos_ratio_Overall = count[0]
    neg_ratio_Overall = count[1]
    pos_count_Overall = pos_ratio1 + pos_ratio2 + pos_ratio3
    neg_count_Overall = neg_ratio1 + neg_ratio2 + neg_ratio3
    total_count_Overall = pos_count_Overall + neg_count_Overall 

    client = pymongo.MongoClient("mongodb+srv://Ravindu:Ravindu1234@cluster01.aco7h.mongodb.net/test?retryWrites=true&w=majority")
    db = client['test']
    collection = db["test"]

    post = {
        "item": item,
        "positive": pos_ratio_Overall,
        "negative": neg_ratio_Overall,
        "positiveCount": pos_count_Overall,
        "negativeCount": neg_count_Overall,
        "featuresCount": 3,
        "totalCount": total_count_Overall,
        "features": {
            "featureOne": {
                "name": featureOne,
                "positive": pos_ratio1,
                "negative": neg_ratio1,
            },
            "featureTwo": {
                "name": featureTwo,
                "positive": pos_ratio2,
                "negative": neg_ratio2,
            },
            "featureThree": {
                "name": featureThree,
                "positive": pos_ratio3,
                "negative": neg_ratio3,
            }
        }

    }
    #"date": datetime.datetime.utcnow()
    collection.insert_one(post).inserted_id
    print(post)
