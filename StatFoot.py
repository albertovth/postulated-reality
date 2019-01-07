# This code is a personal educational exercise, in an attempt to learn and improve my Python-skills.
# The program gathers data on football(soccer) clubs and national teams, and simulates matches.
# The simulations result in frequency distributions.
# These distributions can be used to make informed forecasts.
# Data is collected from FiveThirtyEight's GitHub Soccer API repository.
# The intention is not to reproduce FiveThirtyEight's forecasts, which take in consideration more factors.
# This program makes a forecast for an average match, between two teams.
# FiveThirtyEight's forecasts are more particular: type of tournament, lineups, injuries, etc.
# FiveThirtyEight's data can be substituted by up to date statistics on match results.
# This raw data can be used together with functions to calculate expected goals in an alternative manner.

# Import data from source and create data frames in pandas.
from typing import List, Any

import pandas as pd

spi_global_rankings = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv")
spi_global_rankings.columns = ['rank', 'prev_rank', 'name', 'league', 'off', 'defe', 'spi']
print(spi_global_rankings)

spi_global_rankings_intl = pd.read_csv(
    "https://projects.fivethirtyeight.com/soccer-api/international/spi_global_rankings_intl.csv")
spi_global_rankings_intl.columns = ['rank', 'name', 'confed', 'off', 'defe', 'spi']
print(spi_global_rankings_intl)

# Extract necessary columns

spi_clubs = spi_global_rankings[['name', 'off', 'defe', 'spi']]
spi_national_teams = spi_global_rankings_intl[['name', 'off', 'defe', 'spi']]

# Concatenate data frames to create spi data frame.

spi_frames = [spi_clubs, spi_national_teams]

spi = pd.concat(spi_frames, ignore_index=True)

print(spi)

# Ask user to enter input for home and road team.

home_team_input = str(input('Enter the name of the home team here: '))

road_team_input = str(input('Enter the name of the road team here: '))


def home_team():
    text_home = home_team_input
    print(text_home)

def road_team():
    text_road = road_team_input
    print(text_road)

print("You have registered the following home team: " + home_team_input)
print("You have registered the following road team: " + road_team_input)

# Get logo or flag for each team (not really necessary, but it could be fun to include this in a future GUI).

def club_logo_home():
    try:
        from googlesearch import search
    except:
        ImportError

    query = home_team_input + "FC AND svg AND png AND wikipedia"

    for j in search(query, num=1, stop=1):
        print(j)

    from bs4 import BeautifulSoup as bs
    from urllib.request import urlopen

    html_page = urlopen(j)
    soup = bs(html_page, features='html.parser')
    image_club_logo_ht: List[Any] = []
    for img in soup.findAll('img'):
        image_club_logo_ht.append(img.get('src'))

    return image_club_logo_ht[0]

def country_flag_home():
    try:
        from googlesearch import search
    except:
        ImportError

    query = "'Flag of " + home_team_input + "'" + "File AND svg AND wikipedia"

    for j in search(query, num=1, stop=1):
        print(j)

    from bs4 import BeautifulSoup as bs
    from urllib.request import urlopen

    html_page = urlopen(j)
    soup = bs(html_page, features='html.parser')
    images = []

    for img in soup.findAll('img'):
        images.append(img.get('src'))

    flag_list: List[Any] = [k for k in images if "Flag_of_" in k]

    flag_ht = flag_list[0]

    print(flag_ht)

def print_team_logo_home():
    if spi_global_rankings['name'].str.contains(str(home_team_input)).any():
        return club_logo_home()
    elif spi_global_rankings_intl['name'].str.contains(str(home_team_input)).any():
        return country_flag_home()

print(print_team_logo_home())

def club_logo_road():
    try:
        from googlesearch import search
    except:
        ImportError

    query = road_team_input + "FC AND svg AND png AND wikipedia"

    for j in search(query, num=1, stop=1):
        print(j)

    from bs4 import BeautifulSoup as bs
    from urllib.request import urlopen

    html_page = urlopen(j)
    soup = bs(html_page, features='html.parser')
    image_club_logo_rt: List[Any] = []
    for img in soup.findAll('img'):
        image_club_logo_rt.append(img.get('src'))

    return image_club_logo_rt[0]


def country_flag_road():
    try:
        from googlesearch import search
    except:
        ImportError

    query = "'Flag of " + road_team_input + "'" + " File AND svg AND wikipedia"

    for j in search(query, num=1, stop=1):
        print(j)

    from bs4 import BeautifulSoup as bs
    from urllib.request import urlopen

    html_page = urlopen(j)
    soup = bs(html_page, features='html.parser')
    images = []

    for img in soup.findAll('img'):
        images.append(img.get('src'))

    flag_list: List[Any] = [k for k in images if "Flag_of_" in k]

    flag_rt = flag_list[0]

    print(flag_rt)

def print_team_logo_road():
    if spi_global_rankings['name'].str.contains(str(road_team_input)).any():
        return club_logo_road()
    elif spi_global_rankings_intl['name'].str.contains(str(road_team_input)).any():
        return country_flag_road()

print(print_team_logo_road())

# Extract off and defe for home and road team.

index_home_team = spi.index[spi['name'] == home_team_input]
index_road_team = spi.index[spi['name'] == road_team_input]

home_team_off = spi.at[index_home_team[0], 'off']
home_team_defe = spi.at[index_home_team[0], 'defe']

road_team_off = spi.at[index_road_team[0], 'off']
road_team_defe = spi.at[index_road_team[0], 'defe']

expected_goals_home_team = ((home_team_off) + (road_team_defe)) / 2
expected_goals_road_team = ((road_team_off) + (home_team_defe)) / 2

print('Offensive rating ' + str(home_team_input) + ' = ' + str(home_team_off))
print('Deffensive rating ' + str(home_team_input) + ' = ' + str(home_team_defe))
print('Offensive ratning ' + str(road_team_input) + ' = ' + str(road_team_off))
print('Defensive rating ' + str(road_team_input) + ' = ' + str(road_team_defe))

print('Expected goals ' + str(home_team_input) + ' = ' + str(expected_goals_home_team))
print('Expected goals ' + str(road_team_input) + ' = ' + str(expected_goals_road_team))

# Generate array with 10000 random probabilities for home and visiting team for Monte Carlo simulation.

import random

probability_home = [random.random()]

for x in range(0, 9999):
    probability_home.append(random.random())

print('10 000 random probabilities of possible scores for home team = ', probability_home)

import random

probability_road = [random.random()]

for x in range(0, 9999):
    probability_road.append(random.random())

print('10 000 random probabilities of possible scores for road team = ', probability_road)

# Calculate 10 000 possible scores with inverse poisson function for each team.
# Build results array with this information.

from scipy.stats import poisson

random_scores_home_team = poisson.ppf(probability_home, expected_goals_home_team)
random_scores_road_team = poisson.ppf(probability_road, expected_goals_road_team)
random_scores_match = [str(x[0])+" - " + str(x[1]) for x in zip(random_scores_home_team, random_scores_road_team)]

results = list()

for i, j in zip(random_scores_home_team, random_scores_road_team):
    if i > j:
        results.append("home team wins")
    elif i < j:
        results.append("road team wins")
    else:
        results.append("tie")

print('10 000 random possible results for the match = ', results)

possible_scores_home_team = list(range(0, 11))
possible_scores_road_team = list(range(0, 11))

print('10 000 possible scores for ' + str(home_team_input) + ', ' + 'obtained through inverse poisson, expected goals and random probability = ', random_scores_home_team)
print('10 000 possible scores for ' + str(road_team_input) + ', ' + 'obtained through inverse poisson, expected goals and random probability = ', random_scores_road_team)
print('10 000 possibles scores for the match, obtained through inverse poisson, expected goals and random probabilities = ', random_scores_match)

print('List of possible scores for ' + str(home_team_input) + ' = ', possible_scores_home_team)
print('List of possible scores for ' + str(road_team_input) + ' = ', possible_scores_road_team)

# Calculate goal probability distributions per team.

probability_final_score_home = [((random_scores_home_team == 0).sum()) / 10000,
                                ((random_scores_home_team == 1).sum()) / 10000,
                                ((random_scores_home_team == 2).sum()) / 10000,
                                ((random_scores_home_team == 3).sum()) / 10000,
                                ((random_scores_home_team == 4).sum()) / 10000,
                                ((random_scores_home_team == 5).sum()) / 10000,
                                ((random_scores_home_team == 6).sum()) / 10000,
                                ((random_scores_home_team == 7).sum()) / 10000,
                                ((random_scores_home_team == 8).sum()) / 10000,
                                ((random_scores_home_team == 9).sum()) / 10000,
                                ((random_scores_home_team == 10).sum()) / 10000]

probability_final_score_road = [((random_scores_road_team == 0).sum()) / 10000,
                                ((random_scores_road_team == 1).sum()) / 10000,
                                ((random_scores_road_team == 2).sum()) / 10000,
                                ((random_scores_road_team == 3).sum()) / 10000,
                                ((random_scores_road_team == 4).sum()) / 10000,
                                ((random_scores_road_team == 5).sum()) / 10000,
                                ((random_scores_road_team == 6).sum()) / 10000,
                                ((random_scores_road_team == 7).sum()) / 10000,
                                ((random_scores_road_team == 8).sum()) / 10000,
                                ((random_scores_road_team == 9).sum()) / 10000,
                                ((random_scores_road_team == 10).sum()) / 10000]

probability_final_score_match = [a * b for a, b in zip(probability_home, probability_road)]

print('Probability distribution of final score for ' + str(home_team_input) + " = ", probability_final_score_home)
print('Probability distribution of final score for ' + str(road_team_input) + " = ", probability_final_score_road)

home_team_wins = home_team_input + " wins " + str((results.count("home team wins") / 10000) * 100) + " % of the time"
road_team_wins = road_team_input + " wins " + str((results.count("road team wins") / 10000) * 100) + " % of the time"
tie = "The match ends as a tie " + str((results.count("tie") / 10000) * 100) + " % of the time"

print(home_team_wins)
print(road_team_wins)
print(tie)

# Create diagram for goal probability distributions.

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.plot(possible_scores_home_team, probability_final_score_home, '-o',
         label='Probability of possible final scores ' + home_team_input)
plt.plot(possible_scores_road_team, probability_final_score_road, '-X',
         label='Probability of possible final scores ' + road_team_input)
plt.legend()
plt.show()

# Create pie chart for percentage of wins and ties in simulations.

import matplotlib.pyplot as plt

# Definition of the data that is to be plotted in the diagram.

labels = home_team_input + ' wins', road_team_input + ' wins', 'Tie'
sizes = [results.count("home team wins"), results.count("road team wins"), results.count("tie")]
colors = ['green', 'red', 'gold']

# Pie chart.
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

# Create bins for frequency distribution (Available histogram functions render unorganized results).

list_scores_home = [0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0]

list_scores_road = [0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 0.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0]

# Create list of bins.

possible_scores_match = [str(x[0])+" - " + str(x[1]) for x in zip(list_scores_home, list_scores_road)]

# Construct array of frequency of possible scores.

frequency_of_possible_scores = list()

for i in possible_scores_match:
    result = random_scores_match.count(i)
    frequency_of_possible_scores.append(result)

# Print bins and frequency (visual check).

print('Ordered bin list of possible scores for the match = ', possible_scores_match)
print('Frequency of the ordered bin list of possible scores for the match in Monte Carlo simulation = ', frequency_of_possible_scores)

# Create bar chart to visualize score frequencies in the 10 000 simulations.

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

objects = possible_scores_match
y_pos = np.arange(len(objects))
performance = frequency_of_possible_scores

plt.bar(y_pos, frequency_of_possible_scores, align='center', alpha=0.5)
plt.xticks(y_pos, possible_scores_match, rotation='vertical')
plt.ylabel('Frequency')
plt.title('Frequency of possible scores ' + home_team_input + ' vs. ' + road_team_input)

plt.show()

# Define forecast algorithm in a function.
# Forecast as a winner the team that has more victories and more than 40 % of the victories in the simulations.

def forecast():
    if ((results.count("home team wins")) / 10000) > ((results.count("road team wins")) / 10000) and ((results.count("home team wins")) / 10000) > (0.4): return(str(home_team_input) + " wins")
    elif ((results.count("road team wins")) / 10000) > ((results.count("home team wins")) / 10000) and ((results.count("road team wins")) / 10000) > (0.4): return(str(road_team_input) + " wins")
    else: return("the game ends as a tie")

# Construct forecast scores data frame.

import pandas as pd

Results = results
Scores = random_scores_match

forecast_scores_dataframe=pd.DataFrame(
    {'Results': Results,
     'Scores': Scores})


scores_home_team_wins = forecast_scores_dataframe.loc[forecast_scores_dataframe.Results == "home team wins"]
scores_road_team_wins = forecast_scores_dataframe.loc[forecast_scores_dataframe.Results == "road team wins"]
scores_tie = forecast_scores_dataframe.loc[forecast_scores_dataframe.Results == "tie"]

idxmax_score_home_team_wins = scores_home_team_wins['Scores'].value_counts().idxmax()
idxmax_score_road_team_wins = scores_road_team_wins['Scores'].value_counts().idxmax()
idxmax_score_tie = scores_tie['Scores'].value_counts().idxmax()


# Define score forecast algorithm as a function.
# Note: Score_forecast() and forecast() share a lot. They can be simplified by defining a winner function earlier.

def score_forecast():
    if ((results.count("home team wins")) / 10000) > ((results.count("road team wins")) / 10000) and (
        (results.count("home team wins")) / 10000) > (0.4):
        return(idxmax_score_home_team_wins)
    elif ((results.count("road team wins")) / 10000) > ((results.count("home team wins")) / 10000) and (
        (results.count("road team wins")) / 10000) > (0.4):
        return(idxmax_score_road_team_wins)
    else:
        return(idxmax_score_tie)

# Output textbox with forecast and forecast score.

from tkinter import *

master = Tk()

w = Label(master, text="After 10 000 simulations of the match,\n and considering the latest offensive \nand defensive ratings of the teams,\n the forecast is that " + str(forecast()) + " - " + str(score_forecast()), font=('Times New Roman','46'))
w.pack()

mainloop()

print("After 10 000 simulations of the match, and considering the latest offensive and defensive ratings of the teams, the forecast is that " + str(print(forecast())) + str(print(score_forecast())))




