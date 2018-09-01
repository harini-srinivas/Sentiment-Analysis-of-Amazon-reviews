import sys
import numpy as np
import time
from sklearn.externals import joblib
start = time.clock()
f = open(sys.argv[2], "r")
data = f.readlines()


filename = 'finalized_model.sav'
vectorizer_filename = 'countv.sav'
transformer_filename = 'tfidf.sav'
loaded_model = joblib.load(filename)
loaded_cv = joblib.load(vectorizer_filename)
loaded_tfidf = joblib.load(transformer_filename)

bag_of_words_test = loaded_cv.transform(data)
X_test_tfidf = loaded_tfidf.transform(bag_of_words_test)

result = loaded_model.predict(X_test_tfidf)
result_str = np.array_str(result)
result_str = result_str.strip("[")
result_str = result_str.strip("]")
w = open("output.txt","w")
w.write(result_str)

endtime = time.clock()
print(endtime-start)
