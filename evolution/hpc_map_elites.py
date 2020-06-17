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
                        if (p[1][0] >= q[1][0] and p[1][1] >= q[1][1]) and (p[1][0] > q[1][0] or p[1][1] > q[1][1]):
                            dominated_count += 1
                            dominated_set.append(q)
                        elif (q[1][0] >= p[1][0] and q[1][1] >= p[1][1]) and (q[1][0] > p[1][0] or q[1][1] > p[1][1]):
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
            scores = {}
            keys_found = {}
            j = 0
            for p in front:
                #print(p[1][0])
                #print(p[1][1])
                scores[str(i) + "-" +str(j)] = p[1][0]
                keys_found[str(i) + "-" +str(j)] = p[1][1]
                j += 1
                
            scores = sorted(scores.items(), key=lambda item: item[1])
            #keys_found  = {k: v for k, v in sorted(keys_found.items(), key=lambda item: item[1])}
            keys_found = sorted(keys_found.items(), key=lambda item: item[1])
            crowding_dist_score = {}
            crowding_dist_keys = {}
            crowding_dist = {}
            
            for k in range(len(scores)):
                if k == 0 or k == len(scores)-1:
                    crowding_dist_score[scores[k][0]] = 99999
                else:
                    if scores[len(scores)-1][1] - scores[0][1]  != 0:
                        crowding_dist_score[scores[k][0]] = (scores[k+1][1] - scores[k-1][1])/(scores[len(scores)-1][1] - scores[0][1] )
                    else:
                        crowding_dist_score[scores[k][0]] = 0
                
            for k in range(len(keys_found)):
                if k == 0 or k == len(keys_found)-1:
                    crowding_dist_keys[keys_found[k][0]] = 99999
                else:
                    if keys_found[len(keys_found)-1][1] - keys_found[0][1] != 0:
                        crowding_dist_keys[keys_found[k][0]] = (keys_found[k+1][1] - keys_found[k-1][1])/(keys_found[len(keys_found)-1][1] - keys_found[0][1])
                    else:
                        crowding_dist_keys[keys_found[k][0]] = 0
            #print(crowding_dist_keys)
            #print(crowding_dist_score)
            for j in range(len(front)):
                key_val = str(i) + "-" + str(j)
                if crowding_dist_keys[key_val] == 99999 or crowding_dist_score[key_val] == 99999:
                    crowding_dist[key_val] = 99999
                else:
                    crowding_dist[key_val] = math.sqrt(crowding_dist_keys[key_val]**2 + crowding_dist_score[key_val]**2)
                    
            
            crowding_dists.append(crowding_dist)
            i += 1
        return crowding_dists
    
    def select_new_population(self,fronts,crowding_dists):
        #k = 25
        k = 10
        new_population_len = 40 
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
        