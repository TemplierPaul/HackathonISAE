import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class pipeline():
    def __init__(self, path='Data_ML/MDataFiles_Stage1/', season=2019):
        self.season = season
        self.march_madness = pd.read_csv(path + 'MNCAATourneyCompactResults.csv')
        self.seeds = pd.read_csv(path + 'MNCAATourneySeeds.csv')
        
        seasons = pd.read_csv(path + 'MSeasons.csv')
        self.seasons = list(seasons['Season'].unique())
        
        teams = pd.read_csv(path + 'MTeams.csv')
        self.team_ids = list(teams['TeamID'])
        
        # Match history for training
        self.match_histo = None
        self.results_histo = None
        self.get_history()
        
        
        self.model = None
        self.added_data = []
        self.doubled_data=[]
        
    def get_history(self):
        m = self.march_madness[self.march_madness['Season']<self.season]

        m1 = pd.DataFrame()
        m1['Season'] = m['Season']
        m1['ID_Team1'] = m['WTeamID']
        m1['ID_Team2'] = m['LTeamID']
        m1['Team1_Home'] = (m['WLoc']=='H').astype(int)
        m1['Team2_Home'] = (m['WLoc']=='A').astype(int)
        m1['Team1_Win'] = 1

        m2 = pd.DataFrame()
        m2['Season'] = m['Season']
        m2['ID_Team1'] = m['LTeamID']
        m2['ID_Team2'] = m['WTeamID']
        m2['Team1_Home'] = (m['WLoc']=='A').astype(int)
        m2['Team2_Home'] = (m['WLoc']=='H').astype(int)
        m2['Team1_Win'] = 0

        self.match_histo = pd.concat([m1, m2])
        self.results_histo = self.match_histo['Team1_Win']
        self.match_histo = self.match_histo.drop(columns = ['Team1_Win'])
        print(self.match_histo['Season'].unique())
        return self

    def add_team_data(self, path):
        data_teams = pd.read_csv(path, index_col=0)
        self.match_histo = pd.merge(self.match_histo, data_teams, 
                                  how='left', 
                                  left_on=['Season', 'ID_Team1'], 
                                  right_on=['Season', 'TeamID'])
        self.match_histo = pd.merge(self.match_histo, data_teams, 
                                  how='left', 
                                  left_on=['Season', 'ID_Team2'], 
                                  right_on=['Season', 'TeamID'], 
                                  suffixes=['_Team1', '_Team2'])
        print('History merged with %s' %path)
        self.added_data.append(path)
        self.model = None
        return self
    
    def compute_differences(self, names):
        if type(names)!=list and type(names)!=tuple:
            names = [names]
        for n in names:
            self.match_histo[n + '_diff']=self.match_histo[n+'_Team1'] - self.match_histo[n+'_Team2']
            self.doubled_data.append(n)
            print("Difference computed on %s" %n)
        return self
    
    def train_model(self, season): ## AJOUT
        self.model = RandomForestClassifier(n_estimators=1000)
        self.model.fit(self.match_histo, self.results_histo)
        self.results_histo = pd.DataFrame(self.results_histo) ## AJOUT
        self.match_histo.to_csv('X_' + str(season) + '.csv', index=False) ## AJOUT
        self.results_histo.to_csv('Y_' + str(season) + '.csv', index=False) ## AJOUT
        return self
    
    def predict(self, out='predict.csv'):
        if self.model is None:
            self.train_model()
        f = self.seeds['Season']==(self.season)
        mad_teams = list(self.seeds[f]['TeamID'])

        generated_matches = []
        duos = []
        for i in mad_teams:
            for j in mad_teams:
                if (j, i) not in duos:
                    duos.append((i, j))
                    if i != j:
                        d = {
                            'Season':self.season,
                            'ID_Team1':i,
                            'ID_Team2':j,
                            'Team1_Home':0,
                            'Team2_Home':0
                        }
                        generated_matches.append(d)
        generated_matches = pd.DataFrame(generated_matches)
        for path in self.added_data:
            data_teams = pd.read_csv(path, index_col=0)
            generated_matches = pd.merge(generated_matches, data_teams, 
                                      how='left', 
                                      left_on=['Season', 'ID_Team1'], 
                                      right_on=['Season', 'TeamID'])
            generated_matches = pd.merge(generated_matches, data_teams, 
                                      how='left', 
                                      left_on=['Season', 'ID_Team2'], 
                                      right_on=['Season', 'TeamID'], 
                                      suffixes=['_Team1', '_Team2'])
            print('Generated matches merged with %s' %path)
            
        for n in self.doubled_data:
            generated_matches[n + '_diff']=generated_matches[n+'_Team1'] - generated_matches[n+'_Team2']
        
        # Predict probas
        team1_proba = self.model.predict_proba(generated_matches)
        
        result = pd.DataFrame()
        result[['Année', 'ID_Team1', "ID_Team2"]] = generated_matches[['Season', 'ID_Team1', "ID_Team2"]]
        result['Predic_Team1'] = team1_proba[:,0]
        
        result.to_csv(out, index=False)
        print("Result saved as %s" %out)
        return self