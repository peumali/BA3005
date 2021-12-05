# BA3005
Enhancing Book Recommendation  with the use of Reviews

## Introduction
Recommendation  systems  are  a  major component  in  current e-commerce  websites  and applications. There are many studies carried out to ensure that the best recommendations are provided  to  the  user  and  conversion  rate  is  increased.  These  techniques  usually  utilize historical  transaction  data  and  user  ratings.  While  most  of  such  websites  also  provide  the capability  to  review  the  products  bought  by  the  users,  the  content  of  these  reviews  usually does not play a major role in recommendations made to the users.
Goodreads is the world’s largest website  for readers and book  recommendations. A user can keep track of their reading as well as review, rate and recommend books to other users. Book recommendations  are  also  made  automatically  by  Goodreads  based  on  the  books  a  user  has already read and rated. As a review is much more expressive than a single rating and tends to explain the user’s decision  for  a  rating,  it  is  reasonable  to  expect  that  incorporating  reviews will improve the  recommendation process. This  study attempts to address  this by combining sentiment analysis of the user reviews with the recommendation process in Goodreads.
To  achieve  the  above  goal,  the  constructed  recommender  system  utilizes  LightFM,  a  Python library  facilitating  popular  recommendation  algorithms  for  implicit  and  explicit  feedback. LightFM enables item and user metadata to be incorporated into traditional matrix
factorization  algorithms.  Sentiment  scores  ranging  from  -1  to  1  extracted  from  the  user reviews for each book were utilized as item metadata in the above LightFM model.
This  recommender  system  performed  better  than  pure  collaborative  filtering  algorithms  such as k-nearest neighbor and SVD for the same Goodreads dataset as evinced by the better Area Under the Curve (AUC) score of the LightFM model. The LightFM model reported an AUC score  of  0.884  while  the  K-NN  and  SVD  models  reported  AUC  scores  of  0.61  and  0.6 respectively.  The  final  LightFM  model  also  reported  a  precision@10  value  of  0.67  and recall@10 value of 0.10.

## Data
This study utilizes a publicly available dataset which contains user interaction, user review and book metadata information from Goodreads (Wan & McAuley, 2018). The dataset is categorized into several genres for ease of handling. This study will utilize the data in the Poetry category.
1. https://drive.google.com/uc?id=1H6xUV48D5sa2uSF_BusW-IBJ7PCQZTS1
2. https://drive.google.com/uc?id=17G5_MeSWuhYnD4fGJMvKRSOlBqCCimxJ
3. https://drive.google.com/uc?id=1FVD3LxJXRc5GrKm97LehLgVGbRfF9TyO

## Code

sentiment-analysis.ipynb notebook reads the review data given in json format, removes non-English reviews and calculates sentiment intensities for each review.
sentiment-enhanced-lightfm.ipynb notebook calculate the sentiment intensity for each book. Then using the review data, a LightFM model is created with appropriate parameters.
This model is then used to recommend books to a given user.