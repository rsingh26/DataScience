# Analysis of data of local business using HIVE/PIG
Local Business Data Analysis from Yelp and Google over 10 years (2006-2016)
Local Business Data and the corresponding customer reviews across the U.S are collected using Yelp and Google Local API. The data is analyzed for the popularity of various businesses and the sentiment analysis based on the customer ratings and the reviews.

- Getting Started 
- Data Sets
- Analysis
- Visualizations
- Tutorial 
- License
- Contribution

## Getting started
Tutorial folder contains the project tutorial file with step by step instruction to build this project.
DataSourceLinks.dox provides the two links for business and reviews datasets.
Hive-Query folder contains 1. Final Query - Local Business.txt contains all the queries related to business dataset. 2.Final Query File - Reviews.txt contains all the queries related to reviews dataset.
Visualizations-Images folder contains all the visualizations generated using Tableau and Excel PowerView on the quried data.
## Data Sets
The Local Business dataset collected from Yelp and Google APIs are dated between 2005 and 2016, is rich in information about the local businesses in cities from 14 states in U.S and 4 cities that include: U.K: Edinburgh, Germany: Karlsruhe, Canada: Montreal and Waterloo. The yelp data set is of size 90MB in CSV file format with 334,335 rows and 108 columns. The Google Local reviews dataset is of size 85MB in JASON file format with 117,486 and 10 columns. To be able to use this data in Hadoop, the JSON file was converted to a CSV file using SerDe (Serializer/Deserializer) in Hive. Alternatively, JSON to CSV file format conversion can also be done in PIG using the built-in function JSONLoader().
## Analysis
The clean data is imported into HDFS, exploratory data analysis was performed on the dataset using LOAD and DUMP in Pig to gather some initial facts about our data. For example, the Local Business data includes 2,903,217 entries having attributes such as business id, address, name, longitude, latitude, etc. The Reviews data includes 1,174,850 hive entries having attributes that included business id , user id, review id, stars, review date, and the user comments.
## Visualization
### Video(Youtube) - Geo-spatial representaion of the sentiments
[![Alt text for your video](http://img.youtube.com/vi/OYQ8qrapF2Q/0.jpg)](http://www.youtube.com/watch?v=OYQ8qrapF2Q)
### Most popular sub categories grouped by state in US
![Alt text](https://github.com/shamaahsaa/Local_Business_DataAnalysis/blob/master/Visualizations-Images/1.jpg "")
### Count of reviews for the sub-categories of the Shopping category
![Alt text](https://github.com/shamaahsaa/Local_Business_DataAnalysis/blob/master/Visualizations-Images/2.jpg "Optional Title")
### Percentage of positive, neutral and negative customer reviews for the Services category
![Alt text](https://github.com/shamaahsaa/Local_Business_DataAnalysis/blob/master/Visualizations-Images/3.jpg "Optional Title")
### Businesses categories grouped by city
![Alt text](https://github.com/shamaahsaa/Local_Business_DataAnalysis/blob/master/Visualizations-Images/4.jpg "Optional Title")
### Maximum count of reviews made by individual users
![Alt text](https://github.com/shamaahsaa/Local_Business_DataAnalysis/blob/master/Visualizations-Images/5.jpg "Optional Title")
### Sentiment Analysis based on the customer reviews 
![Alt text](https://github.com/shamaahsaa/Local_Business_DataAnalysis/blob/master/Visualizations-Images/6.jpg "Optional Title")

## Tutorial
Tutorial pdf file: https://csula-my.sharepoint.com/personal/rsingh26_calstatela_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=g%2faLqtF1Uqw5KqJZ6W7w%2b4AQ8mY%2f6qdDm5sMhgOztxk%3d&docid=2_1ffdc4c596d014906ac4fabbf01a82e69&rev=1
This tutorial also available in this GitHub.
## Contribution
Local Business Data Set and its's analysis is put together by the graduate students - Hemamalini Madhanguru (hmadhan@calstatela.edu), Mahsa Tayer Farahani (mtayerf@calstatela.edu), Ruchi Singh (rsingh26@calstatela.edu), Yashaswi Ananth (yananth@calstatela.edu) from the Dept. of Computer Information Systems, California State University Los Angeles.

## License
### MIT LICENSE
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



