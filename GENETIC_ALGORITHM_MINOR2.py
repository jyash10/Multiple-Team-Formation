from itertools import islice
import random
import time
import matplotlib.pyplot as plt
import copy
import csv


def initialization(expert_size,pop_size, ch_size):
    chrom_size = ch_size  #this functiion is used to generate initial   
    initial_pop = []      #population
    for _ in range(pop_size):
        list = random.sample(range(1, expert_size+1), chrom_size)
        initial_pop.append(list)
    return initial_pop


def fitnessfunc1(chromosome, sociomat, projects_size):  # for sociometric matrix
    mylist = copy.deepcopy(chromosome)
    seclist = [projects_size[i] for i in range(0,len(projects_size))]
    it = iter(mylist)
    teams = [list(islice(it, 0, i)) for i in seclist]
    sum = 0
    for k in range(0, len(teams)):
        list1 = teams[k]
        for i in range(0, len(teams[k])):
            # for each team we have to calculate score
            for j in range(0, len(teams[k])):
                if(i != j):
                    sum += sociomat[list1[j]-1][list1[i]-1]
    return sum


def fitnessfunc2(req_mat, skill_mat, projects_size, chromosome):
    mylist = copy.deepcopy(chromosome)
    seclist = [projects_size[i] for i in range(0,len(projects_size))]
    it = iter(mylist)
    score = 0
    teams = [list(islice(it, 0, i)) for i in seclist]
    project_index = 0
    for k in range(0, len(teams)):
        list1 = teams[k]
        i = 0
        team_len = len(list1)
        skill_index = 0
        while(i < team_len):
            skill_size = req_mat[project_index][skill_index]
            j = 0
            while(j < skill_size):
                score += skill_mat[list1[i]-1][skill_index]
                score+=skill_mat[list1[i]-1][6]
                score+=skill_mat[list1[i]-1][7]
                j = j+1
                i = i+1
            skill_index = skill_index+1
        project_index = project_index+1
    return score


def combined_fitness_func(req_mat, skill_mat, projects_size, chromosome,sociomat):
    score1=fitnessfunc1(chromosome,sociomat,projects_size)
    score2=fitnessfunc2(req_mat,skill_mat,projects_size,chromosome)
    return score1+score2   #this function simply adds the values of fitnessfunc1 and fitnessfunc2


def requirement_matrix_gen(): #GENERATES THE REQUIREMNT MATRIX
                              #BASICALLY IT DEFINES THE STRUCTURE OF THE CHROMOSOME
    no_of_projects = 9
    no_of_skills = 8
    requirement_matrix = [[0,3,0,1,0,1,0,0],[1,1,0,1,0,0,2,0],[1,0,1,1,0,1,0,1],[1,0,0,1,2,0,0,1],[0,0,2,1,1,1,0,0],[0,3,0,0,1,0,0,1],[0,0,0,1,0,1,2,1],[0,2,0,0,1,0,2,0],[0,0,0,4,0,0,1,0]]
    all_project_size = []  # stores each project size 4,2,2,2
    chromosome_size = 0     # stores total sum 4+2+1
    for row in requirement_matrix:
        proj_s = 0
        for item in row:
            chromosome_size += item
            proj_s += item
        all_project_size.append(proj_s)
    return requirement_matrix, all_project_size, chromosome_size


def crossover_two(parent_1, parent_2, crossover_rate):  # two point crossover
    count = 0
    while(count<10):        #LOOP IS USED TO AVOID GENERATING THE SAME CHROMOSOME AGAIN AND AGAIN
     point_1, point_2 = random.sample(range(1, len(parent_1)-1), 2)
     begin = min(point_1, point_2)
     end = max(point_1, point_2)
     child_1 = copy.deepcopy(parent_2[begin:end+1])
     child_2 = copy.deepcopy(parent_1[begin:end+1])
     
     flag=False
     if(random.random() > crossover_rate):
        return parent_1, parent_2
     d1 = dict() #THIS IS USED TO KEEP A CHECK ON THE PEOPLE THAT ARE ALREADY PRESENT IN THE CHROMOSOME
     f = 1       #SINCE NO TWO PEOPLE CAN  WORK ON TWO DIFFERENT PROJECTS AT THE SAME TIME
     for item in parent_1:
        d1[item] = 1
     for item in child_1:
        if(d1.get(item) != None):
            f = 0
            break
     if(f == 0):
        child_1 = copy.deepcopy(parent_1)
     else:
        child_1 = copy.deepcopy(parent_1)
        for x in range(begin, end+1):
            child_1[x] = parent_2[x]
        flag=True
     d1 = dict()
     f = 1
     for item in parent_2:
        d1[item] = 1
     for item in child_2:
        if(d1.get(item) != None):
            f = 0
            break
     if(f == 0):
        child_2 = copy.deepcopy(parent_2)
     else:
        child_2 = copy.deepcopy(parent_2)
        for x in range(begin, end+1):
            child_2[x] = parent_1[x]
        flag=True
     if(flag==True):
        break
     count=count+1
    return child_1, child_2


def mutation(chromosome, expert_size, mutation_rate): # THIS FUNCTION RANDOMLY PICKS A PERSON IN THE PROJECT
                                                      # AND REPLACES HIM/HER WITH ANOTHER PERSON 
    new_chromosome=copy.deepcopy(chromosome)
    chrom_size = len(new_chromosome)
    mutation_index = random.randint(0, chrom_size-1) 
    ch = new_chromosome[mutation_index]
    new_val = random.randint(1, pop_size)
    count = 0
    if(random.random() < mutation_rate):
        d = dict()
        i = 0
        while(i < expert_size+1):
            d[i] = 0
            i += 1
        i = 0
        while(i < len(new_chromosome)):
            d[new_chromosome[i]] = 1
            i += 1
        while(count < 10 and d.get(new_val)):
            new_val = random.randint(1, expert_size)
            count += 1

        if(count < 10):
            new_chromosome[mutation_index] = new_val
    return new_chromosome


def selection(population, sociomat, projects_size):  # tournament selection on the basis of fitnessfunc1 only
    index_1, index_2, index_3, index_4 = random.sample(
        range(0, len(population)), 4)  # random 4 tickets

    # create candidate chromosomes based on ticket numbers
    candidate_1 = population[index_1]
    candidate_2 = population[index_2]
    candidate_3 = population[index_3]
    candidate_4 = population[index_4]

    # select the winner according to their costs
    if fitnessfunc1(candidate_1, sociomat, projects_size) > fitnessfunc1(candidate_2, sociomat, projects_size):
        winner = candidate_1
        winner_index = index_1
    else:
        winner = candidate_2
        winner_index = index_2

    if fitnessfunc1(candidate_3, sociomat, projects_size) > fitnessfunc1(winner, sociomat, projects_size):
        winner = candidate_3
        winner_index = index_3
    if fitnessfunc1(candidate_4, sociomat, projects_size) > fitnessfunc1(winner, sociomat, projects_size):
        winner = candidate_4
        winner_index = index_4

    return winner,winner_index  # winner = chromosome


def selection_2(population, skill_mat, req_mat, projects_size):  
    # tournament selection on the basis of fitnessfunc2 only
    index_1, index_2, index_3, index_4 = random.sample(
        range(0, len(population)), 4)  # random 4 tickets

    # create candidate chromosomes based on ticket numbers
    candidate_1 = population[index_1]
    candidate_2 = population[index_2]
    candidate_3 = population[index_3]
    candidate_4 = population[index_4]

    # select the winner according to their costs
    if fitnessfunc2(req_mat, skill_mat, projects_size, candidate_1) > fitnessfunc2(req_mat, skill_mat, projects_size, candidate_2):
        winner = candidate_1
        winner_index = index_1
    else:
        winner = candidate_2
        winner_index = index_2

    if fitnessfunc2(req_mat, skill_mat, projects_size, candidate_3) > fitnessfunc2(req_mat, skill_mat, projects_size, winner):
        winner = candidate_3
        winner_index = index_3
    if fitnessfunc2(req_mat, skill_mat, projects_size, candidate_4) > fitnessfunc2(req_mat, skill_mat, projects_size, winner):
        winner = candidate_4
        winner_index = index_4

    return winner,winner_index  # winner = chromosome


def selection_3(population, skill_mat, req_mat, projects_size,sociomat):  # tournament selection
    # tournament selection on the basis of combined_fitness_func(func1+func2) only
    index_1, index_2, index_3, index_4 = random.sample(
        range(0, len(population)), 4)  # random 4 tickets
 
    # create candidate chromosomes based on ticket numbers
    candidate_1 = population[index_1]
    candidate_2 = population[index_2]
    candidate_3 = population[index_3]
    candidate_4 = population[index_4]

    # select the winner according to their costs
    if combined_fitness_func(req_mat,skill_mat, projects_size,candidate_1,sociomat) > combined_fitness_func(req_mat,skill_mat, projects_size,candidate_2,sociomat):
        winner = candidate_1
        winner_index = index_1
    else:
        winner = candidate_2
        winner_index = index_2

    if combined_fitness_func(req_mat,skill_mat, projects_size,candidate_3,sociomat) > combined_fitness_func(req_mat,skill_mat, projects_size,winner,sociomat):
        winner = candidate_3
        winner_index = index_3
    if combined_fitness_func(req_mat,skill_mat, projects_size,candidate_4,sociomat) > combined_fitness_func(req_mat,skill_mat, projects_size,winner,sociomat):
        winner = candidate_4
        winner_index = index_4

    return winner,winner_index  # winner = chromosome



def run_evolution_3(population, sociomat, projects_size,expert_size,req_mat,skill_mat):
     #this fuction is working on the combined values of the 
     # fitnessfunc1+fitnessfunc2
    start = time.time()
    fitness_values1 = [] #for fitnessfunc1
    fitness_values2 = [] #for fitnessfunc2
    fitness_values3 = [] #for fitnessfunc3(fitnessfunc1+fitnessfunc2)
    crossover_rate = 0.7
    mutation_rate = 0.3
    population_2=copy.deepcopy(population) #for fitnessfunc1
    population_3=copy.deepcopy(population) #for fitnessfunc2
    avgfitness=[]
    
    gen_size = 100 #Geneartion size
    for i in range(gen_size):
        print("\nIn gen:",i)
        #sorts the population on the basis of decraesing order of their fitness values of combined_fitness_func
        population = sorted(population, key=lambda genome: combined_fitness_func(
            req_mat,skill_mat, projects_size,genome,sociomat), reverse=True)
        #sorts the population on the basis of decraesing order of their fitness values of function1
        population_2= sorted(population_2, key=lambda genome: fitnessfunc1(
            genome, sociomat, projects_size), reverse=True)
        #sorts the population on the basis of decraesing order of their fitness values of function1
        population_3= sorted(population_3,
                            key=lambda genome: fitnessfunc2(req_mat, skill_mat, projects_size, genome), reverse=True)
        print(population)
        sum=0
        #used to calcualte the average of the fitness values of all the chromosomes in each generation
        for k in range(len(population)): 
            sum+=combined_fitness_func(
            req_mat,skill_mat, projects_size,population[k],sociomat)
        avgfitness.append(sum/(len(population)))

        next_generation = [] #stores next generation using combined_fitness_func1
        next_gen2=[]         #stores next generation using fitnessfunc1
        next_gen3=[]         #stores next generation using fitnessfunc2
        #stores the fitness value(combined_fitness value) of the best chromosome in each generation
        fitness_values3.append(combined_fitness_func(
            req_mat,skill_mat, projects_size,population[0],sociomat))
        #stores the fitness value(fitnessfunc1) of the best chromosome in each generation
        fitness_values1.append(fitnessfunc1(population_2[0], sociomat, projects_size))
        #stores the fitness value(fitnessfunc2) of the best chromosome in each generation
        fitness_values2.append(fitnessfunc2(req_mat,skill_mat,projects_size,population_3[0]))
        for j in range(int(len(population)/2)-1):
            offspring_a=[]
            offspring_b=[]
            parent_1=[]
            parent_2=[]
            parent_1=-1
            parent_1,parent_1index = selection_3(population,skill_mat ,req_mat,projects_size,sociomat)
            new_pop=copy.deepcopy(population)
            new_pop.remove(new_pop[parent_1index])
            parent_2,parent_2index = selection_3(new_pop,skill_mat ,req_mat,projects_size,sociomat)
            print("\nParent1:",parent_1,"Parent2:",parent_2,"\n")
            offspring_a, offspring_b = crossover_two(
                parent_1, parent_2, crossover_rate)
            print(offspring_a," -",offspring_b)
            offspring_a = mutation(offspring_a, expert_size,mutation_rate)
            offspring_b = mutation(offspring_b, expert_size,mutation_rate)
            next_generation.append(offspring_a)
            next_generation.append(offspring_b)
            next_gen2.append(offspring_b)
            next_gen2.append(offspring_b)
            next_gen3.append(offspring_b)
            next_gen3.append(offspring_b)
        next_generation.append(population[0])
        next_generation.append(population[1])
        next_gen2.append(population_2[0])
        next_gen2.append(population_2[1])
        next_gen3.append(population_3[0])
        next_gen3.append(population_3[1])
        population =copy.deepcopy(next_generation)
        population_2 =copy.deepcopy(next_gen2)
        population_3 =copy.deepcopy(next_gen3)
    end=time.time()
    population=[]
    population = sorted(next_generation, key=lambda genome: combined_fitness_func(
            req_mat,skill_mat, projects_size,genome,sociomat), reverse=True)
    population_2=[]
    population_2= sorted(next_gen2, key=lambda genome: fitnessfunc1(
            genome, sociomat, projects_size), reverse=True)
        #sorts the population on the basis of decraesing order of their fitness values of function1
    population_3=[]
    population_3= sorted(next_gen3,
                            key=lambda genome: fitnessfunc2(req_mat, skill_mat, projects_size, genome), reverse=True)
    print("Fittest Chromosome with respect to Combined Fitness",population[0])
    print("Fittest Value with respect to combined fitness",combined_fitness_func(req_mat,skill_mat,projects_size,population[0],sociomat))
    print("Fittest Chromosome with respect to Fitness function1",population_2[0])
    print("Fitness Value with respect to Fitness Function1",fitnessfunc1(population_2[0],sociomat,projects_size))
    print("Fittest Chromosome with respect to Fitness function2",population_3[0])
    print("Fitness Value with respect to Fitness Function2",fitnessfunc2(req_mat,skill_mat,projects_size,population_3[0]))
    print("Total Time Taken",end-start)
    

    
    x = [i for i in range(1, gen_size+1)]
    plt.plot(x, fitness_values1)
    plt.xlabel('Genetration number')
    # naming the y axis
    plt.ylabel('Fitness_Function1')
    # giving a title to my graph
    plt.title('SocioRelation')
    # function to show the plot
    plt.show()

    x = [i for i in range(1, gen_size+1)]
    plt.plot(x, fitness_values2)
    plt.xlabel('Generation number')
    # naming the y axis
    plt.ylabel('FitnessFunction2')
    # giving a title to my graph
    plt.title('SKILL')
    # function to show the plot
    plt.show()

    x = [i for i in range(1, gen_size+1)]
    plt.plot(x, fitness_values3)
    plt.xlabel('Genetration number')
    # naming the y axis
    plt.ylabel('combined_fitness_values')
    # giving a title to my graph
    plt.title('Generation Number Vs Combined_fitness_values ')
    # function to show the plot
    plt.show()

    x = [i for i in range(1, gen_size+1)]
    plt.plot(x, avgfitness)
    plt.xlabel('genetration number')
    # naming the y axis
    plt.ylabel('AverageFitnessValues')
    # giving a title to my graph
    plt.title('average efficiency')
    # function to show the plot
    plt.show()


pop_size = 30  #Population size
expert_size=100  
with open('SOCIOMETRIC_MATRIX.csv',encoding='utf-8-sig')as csvfile:
    reader=csv.reader(csvfile)  #to get values from csv file and store it in sociomat
    sociomat=[]
    for row in reader:
      # print(row)
      sociomat.append(row)
for i in range(100):
    for j in range(100):
        sociomat[i][j]=int(sociomat[i][j])
print("Sociometric matrix\n",sociomat)

req_mat, all_project_size, chromosome_size = requirement_matrix_gen()
population = initialization(expert_size,pop_size, chromosome_size) #generates initial population
print(population)


with open('EFFICIENCY_MATRIX(1).csv',encoding='utf-8-sig')as csvfile:
    Arrayskill_result=csv.reader(csvfile)
    skill_mat=[] #stores the rating of each individual in different skills
    for row in Arrayskill_result:
      # print(row)
      skill_mat.append(row)
for i in range(100):
    for j in range(8): #8 skills are taken into account
        skill_mat[i][j]=int(skill_mat[i][j])
print("Skill matrix\n",skill_mat)

run_evolution_3(population,sociomat,all_project_size,expert_size,req_mat,skill_mat)

