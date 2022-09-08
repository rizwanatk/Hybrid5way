
"""
Thsi code use VGG16 pretrain weights
2- extract features
3- Use PCA
4- USE Random forest RF
5- Add layers and freeze blocks to fine tune.

"""
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Model, Sequential
import seaborn as sns
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input,  BatchNormalization, Dropout

from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix,  classification_report, accuracy_score, recall_score, precision_score, f1_score

from tensorflow.keras.optimizers import SGD

# Read input images and assign labels based on folder names
print(os.listdir("images/"))

SIZE = 224  # Resize images

# Capture training data and labels into respective lists
train_images = []
train_labels = []

for directory_path in glob.glob("images/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

# Convert lists to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Capture test/validation data and labels into respective lists

test_images = []
test_labels = []
for directory_path in glob.glob("images/validation/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

# Convert lists to arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels from text to integers.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Scale pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One hot encode y values for neural network. Not needed for Random Forest
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#############################
# Load VGG model with imagenet trained weights and without classifier/fully connected layers
# We will use this as feature extractor.
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
#for layer in VGG_model.layers:
for layer in VGG_model.layers[0:-10]:
#for layer in VGG_model.layers:
    layer.trainable = False
VGG_model.summary()  # Trainable parameters will be 0

# Now, let us extract features using VGG imagenet weights
# Train features
#train_feature_extractor = VGG_model.predict(x_train)
#train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
# test features
#test_feature_extractor = VGG_model.predict(x_test)
#test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

# Reduce dimensions using PCA
#from sklearn.decomposition import PCA

# First verfiy the ideal number of PCA components to not lose much information.
# Try to retain 90% information, so look where the curve starts to flatten.
# Remember that the n_components must be lower than the number of rows or columns (features)
#pca_test = PCA(n_components=10)  #
#figp = plt.figure(figsize=(8, 6))
#pca_test.fit(train_features)
#plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
#plt.xlabel("Number of components")
#plt.ylabel("Cum variance")
#figp.savefig('confusion_matrix')

# Pick the optimal number of components. This is how many features we will have
# for our machine learning
#n_PCA_components = 10
#pca = PCA(n_components=n_PCA_components)
#train_PCA = pca.fit_transform(train_features)
#test_PCA = pca.transform(test_features)  # Make sure you are just transforming, not fitting.

# If we want 90% information captured we can also try ...
# pca=PCA(0.9)
# principalComponents = pca.fit_transform(X_for_RF)

############## Neural Network Approach ##################

##Add hidden dense layers and final output/classifier layer.
x = Flatten()(VGG_model.output)
x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
#x = Maxpooling()(x)
#x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
#x = Dropout(0.50)(x)
x= Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
#x = Dense(128, activation='relu')(x)
#x = BatchNormalization()(x)

#x = Dropout(0.5)(x)
#x = Dense(128, activation='relu')(x)
#x = Maxpooling()(x)
#x = BatchNormalization()(x)
#x = Dense(128, activation='relu')(x)
#x = Dense(128, activation='relu')(x)
#x = Dense(128, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = BatchNormalization()(x)
#x = Dense(64, activation='relu')(x)
#x = Dropout(0.5)(x)
#output = Dense(5, activation='softmax')(x+x)

# x = Dense(1000, activation='relu')(x)
prediction = Dense(5, activation='softmax')(x)

# create a model object
model = Model(inputs=VGG_model.input, outputs=prediction)
model = Model(inputs=VGG_model.input, outputs=prediction)

print(model.summary())
#SGD = sgd(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

import datetime

start = datetime.datetime.now()
# Fit the model. Do not forget to use on-hot-encoded Y values.
h = model.fit(x_train, y_train_one_hot, epochs=1000, verbose=1)

end = datetime.datetime.now()
print("Total execution time is: ", end - start)

##Predict on test dataset
predict_test = model.predict(x_test)
predict_test = np.argmax(predict_test, axis=1)
predict_test = le.inverse_transform(predict_test)
#
h.history['categorical_accuracy']
Avg_Acc = np.mean(h.history['categorical_accuracy'])
print(Avg_Acc)

##Print overall accuracy
from sklearn import metrics
#print("Accuracy = ", metrics.accuracy_score(test_labels, predict_test))
# Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predict_test)
#print(cm)
#sns.heatmap(cm, annot=True)
#####################################
#cm = confusion_matrix(y_pred,y_test)
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot()
#plt.matshow(cm)
#plt.title('Confusion Matrix for Multiple Classes Using VGG+RF')
#plt.colorbar()
sns.set(font_scale=1.0)
sns.heatmap(cm, annot=True, fmt='g', cmap="crest", ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels', fontsize=15);ax.set_ylabel('True labels', fontsize=15);
ax.set_title('Confusion Matrix', fontsize=15);
ax.xaxis.set_ticklabels(['AD', 'EMCI','MCI', 'LMCI', 'NC'], fontsize=15); ax.yaxis.set_ticklabels(['AD', 'EMCI','MCI', 'LMCI', 'NC'], fontsize=15);
#ax.xaxis.set_ticklabels(['AD',  'NC','MCI', 'LMCI'], fontsize=15); ax.yaxis.set_ticklabels(['AD',  'NC','MCI', 'LMCI'], fontsize=15);
#ax.xaxis.set_ticklabels(['AD',  'NC'], fontsize=15); ax.yaxis.set_ticklabels(['NC',  'AD'], fontsize=15);
fig.savefig('confusion_matrix')

loss = h.history['loss']
val_loss = h.history['loss']
epochs = range(1, len(loss) + 1)
fig1= plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'g', label='Validation loss')
#plt.title('Training and validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
fig1.savefig('LossValidation_loss')

# fit the model
# loss
#fig1 = plt.figure(figsize=(8, 6))
#plt.plot(h.history['val_loss'], label='train loss')
#plt.plot(h.history['val_loss'], label='val loss')
#plt.legend()
#plt.show()
#plt.savefig1('LossValidation_loss.png')

# accuracies
fig2 = plt.figure()
#plt.plot(h.history['categorical_accuracy'], label='train accuracy')
plt.plot(h.history['categorical_accuracy'], label='val accuracy')
plt.legend()
plt.show()
fig2.savefig('AccVal_acc')

fig2 = plt.figure()
#plt.plot(h.history['categorical_accuracy'], label='train accuracy')
plt.plot(h.history['categorical_accuracy'], label='test accuracy')
plt.legend()
plt.show()
fig2.savefig('testAcc_acc')

model.save('facefeatures_new_model.h5')

#model.evaluate(x_test, y_test)
#y_pred = model.predict(x_test)
#y_pred = np.argmax(y_pred, axis = 1)
#print(classification_report(y_pred, y_test))
###F1 score
#F1 =  f1_score(y_test, y_pred, average = 'micro')
### Recall
#Recall = recall_score(y_test, y_pred, average = 'weighted')
###Precision
#Prec = precision_score(y_test, y_pred, average = 'micro')


# Check results on a few select images
#n = np.random.randint(0, x_test.shape[0])
#img = x_test[n]
#plt.imshow(img)

#input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
#input_img_feature = VGG_model.predict(input_img)
#input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
#input_img_PCA = pca.transform(input_img_features)
#prediction_img = model.predict(input_img)
#prediction_img = np.argmax(prediction_img, axis=1)
#prediction_img = le.inverse_transform(prediction_img)  # Reverse the label encoder to original name
#print("The prediction for this image is: ", prediction_img)
#print("The actual label for this image is: ", test_labels[n])


import tensorflow as tf

from keras.models import load_model

#model.save('AD-facefeatures_new_model.h5')
"""
############################################################
#RANDOM FOREST implementation (Uncomment to run this part)

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 80, random_state = 42)

# Train the model on training data
RF_model.fit(train_PCA, y_train) #For sklearn no one hot encoding

#Send test data through same feature extractor process
#X_test_feature = VGG_model.predict(x_test)
#X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(test_PCA)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)

"""




