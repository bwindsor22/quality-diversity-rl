import logging

from evolution.map_elites import MapElites
import math
import random


class HPCMapElites(MapElites):

    def next_model(self):
        if len(self.population) == 0:
            self.model.__init__(*self.init_model)
            model_state = self.model.state_dict()
        else:
            model_state = self.random_variation()
        return model_state
    
    def is_dominant(self,a,b):
        is_not_less_than = True
        has_one_better_objective = False
        for j in range(len(a[1])):
            if a[1][j] < b[1][j]:
                is_not_less_than = False
            if a[1][j] > b[1][j]:
                has_one_better_objective = True
        
        if has_one_better_objective == True and is_not_less_than == True:
            return True
        else:
            return False
        
    
    def non_dominated_sort(self):
        unsorted_population = self.population
        fronts = []
        
        while(len(unsorted_population) != 0):
            front = []
            new_unsorted = []
            for p in unsorted_population:
                dominated_count = 0
                dominated_set = []
                dominated_by_count = 0
                for q in unsorted_population:
                    #if p != q:
                    if p is not q:
                        if self.is_dominant(p,q) == True:
                            dominated_count += 1
                            dominated_set.append(q)
                        elif self.is_dominant(q,p):
                            dominated_by_count +=1
                    
                if(dominated_by_count ==0):
                    front.append(p)
                    #Trouble with list.remove with tensor at index 0 of tuples
                    #unsorted_population.remove(p)
                    #print(p[1][0])
                    #print(p[1][1])
                else:
                    new_unsorted.append(p)
            if front == []:
                break
            fronts.append(front)
            unsorted_population = new_unsorted
            #print("Unsorted pop ",len(unsorted_population))
            #print("Front ",len(front))
        #rank is i+1 for ith front
        return fronts
        
        
    
    def crowding_distance(self, fronts):
        crowding_dists = []
        i = 0
        for front in fronts:
            #Use front number and order in front to come up with dictionary key
            scores = []
            j = 0            
            #Get number of scores in fitness function
            for i in range(len(front[0][1])):
                temp = {}
                scores.append(temp)
            #Uniquely identify elements in fronts based on front index and position in front
            for p in front:
                j += 1
                for k in range(len(p[1])):
                    scores[k][str(i) + "-" + str(j)] = p[1][k]

            crowding_dist = {}
            crowding_dists_along_objectives = []
            sorted_scores = []
            
            for score in scores:
                sorted_scores.append(sorted(score.items(),key = lambda item: item[1]))
                
            for objective in sorted_scores:
                temp = {}
                for k in range(len(objective)):
                    if k ==0 or k == len(objective)-1:
                        temp[objective[k][0]] = 99999
                    else:
                        if objective[len(objective)-1][k] - objective[0][k] != 0:
                            temp[objective[k][0]] = (objective[k+1][1] - objective[k-1][1])/(objective[len(objective)-1][1] - objective[0][1] )
                        else:
                            temp[objective[k][0]] = 0
                crowding_dists_along_objectives.append(temp)
                        
            ##in progress Start from here
            #print(crowding_dist_keys)
            #print(crowding_dist_score)
            '''
            for j in range(len(front)):
                key_val = str(i) + "-" + str(j)
                if crowding_dist_keys[key_val] == 99999 or crowding_dist_score[key_val] == 99999:
                    crowding_dist[key_val] = 99999
                else:
                    crowding_dist[key_val] = math.sqrt(crowding_dist_keys[key_val]**2 + crowding_dist_score[key_val]**2)
            '''
            for j in range(len(front)):
                key_val = str(i) + "-" + str(j)
                is_endpoint = False
                for k in range(len(sorted_scores)):
                    if crowding_dists_along_objectives[k][key_val] == 99999:
                        crowding_dist[key_val] = 99999
                        is_endpoint = True
                        break
                    else:
                        crowding_dist[key_val] += (crowding_dists_along_objectives[k][key_val])**2
                
                if is_endpoint == False:
                    crowding_dist[key_val] = math.sqrt(crowding_dist[key_val])
                
                
            
            crowding_dists.append(crowding_dist)
            i += 1
        return crowding_dists
    
    def select_new_population(self,fronts,crowding_dists):
        #k = 25
        k = 20
        new_population_len = 400 
        if self.population == 0:
            for front in fronts:
                for p in front:
                    self.population.append(p)
            return
        
        new_population = []
        while(len(new_population) < new_population_len):
            max_front_index = random.randint(0,len(fronts)-1)
            max_individual_index = random.randint(0,len(fronts[max_front_index])-1)
            max_solution = fronts[max_front_index][max_individual_index]
            for i in range(0,k):
                front_index = random.randint(0,len(fronts)-1)
                individual_index = random.randint(0,len(fronts[front_index])-1)
                solution = fronts[front_index][individual_index]
                
                if front_index < max_front_index:
                    
                    max_front_index = front_index
                    max_individual_index = individual_index
                    max_solution = fronts[front_index][individual_index]

                elif front_index == max_front_index:
                    
                    if crowding_dists[front_index][str(front_index) + "-" + str(individual_index)] > crowding_dists[max_front_index][str(max_front_index) + "-" + str(max_individual_index)]:
                        max_front_index = front_index
                        max_individual_index = individual_index
                        max_solution = fronts[front_index][individual_index]                        
                    
            new_population.append(max_solution)
        
        self.population = new_population
        
                
    def update_result(self, network, feature, fitness):
        #logging.info('Updating feature {}, performance {}'.format(feature, fitness))
        '''
        if self.cmame:
            self.emitters.tell(feature, network, fitness)
        elif feature not in self.performances or self.performances[feature] < fitness:
            logging.info('Found better performance for feature: {}, new score: {}'.format(feature, fitness))
            self.performances[feature] = fitness
            self.solutions[feature] = network
        logging.info('updated map elites with result')
        '''
        #Replace or add?
        #self.population = []
        self.population.append((network,fitness,feature))
        
        #Organize by rank
        #Find Pareto Front
        #Update rank 
        #Update pop 
        #Update fitness
        
