import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import DistanceMetric

app = Flask(__name__)

# Load the crime data from the CSV file
crime_data = pd.read_csv('crime_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract the latitude and longitude from the form data
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        # Find the 10 closest locations to the input latitude and longitude
        dist = DistanceMetric.get_metric('haversine')
        distances = dist.pairwise(crime_data[['latitude', 'longitude']], [(latitude, longitude)])[:, 0]
        crime_data['distance'] = distances
        closest_locations = crime_data.sort_values('distance').iloc[:10]

        # Train a linear regression model on the closest locations
        model = LinearRegression()
        model.fit(pd.DataFrame(closest_locations, columns=['latitude', 'longitude']), closest_locations['crime_rate'])

        # Predict the crime rate for the input location
        crime_rate = round(model.predict([[latitude, longitude]])[0], 2)

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.title('Crime Rate by Location in Kerala')
        plt.scatter(crime_data['longitude'], crime_data['latitude'], c=crime_data['crime_rate'], cmap='plasma')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Add a green marker for the input location with the predicted crime rate
        plt.scatter(longitude, latitude, c='green', s=100, label=f'Predicted crime rate: {crime_rate}')
        plt.legend()

        # Save the plot to a file
        plt.savefig('static/scatter_plot.png')

        # Create the bar chart
        plt.figure(figsize=(8, 6))
        plt.title('Crime Rate by Location: Nearest 10 Locations')
        bars = plt.bar(range(len(closest_locations)), closest_locations['crime_rate'], color='purple')
        plt.xticks(range(len(closest_locations)), closest_locations.index)
        plt.xlabel('Location')
        plt.ylabel('Crime Rate')

        # Add a legend to explicitly show which bar represents your location
        bars[0].set_color('green')
        plt.legend([bars[0]], ['Your location'])

        # Save the plot to a file
        plt.savefig('static/bar_chart.png')

        # Render the results page with the plots
        return render_template('result.html', crime_rate=crime_rate)

    # If the method is GET, render the input form page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
