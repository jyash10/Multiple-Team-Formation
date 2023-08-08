#Importing required modules
import math
import random
import matplotlib.pyplot as plt
from itertools import islice
import time
import copy
import csv
from statistics import mean

def initialization(expert_size,pop_size, ch_size):
    chrom_size = ch_size  # popsize is diffrent from experts size
    initial_pop = []
    for _ in range(pop_size):
        list = random.sample(range(1, expert_size+1), chrom_size)
        initial_pop.append(list)
    return initial_pop

def requirement_matrix_gen():
    no_of_projects = 6
    no_of_skills = 8
    requirement_matrix =[[0,3,0,1,0,1,0,0],[1,1,0,1,0,0,2,0],[1,0,1,1,0,1,0,1]]
    chromosome_size = 0
    all_project_size = []
    chromosome_size = 0
    for row in requirement_matrix:
        proj_s = 0
        for item in row:
            chromosome_size += item
            proj_s += item
        all_project_size.append(proj_s)
    return requirement_matrix, all_project_size, chromosome_size

#First function to optimize
def function1(chromosome, sociomat, projects_size):  # for sociometric matrix
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


#Second function to optimize

def function2(req_mat, skill_mat, projects_size, chromosome):
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
#Function to find index of list

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))] #it stores a particular chromosome is dominating other chromosome in both the values
    
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))] #it tells 0-means the solution is best
                                               #1- means second best solution and so on

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0 #it stores the number by which a chromosome gets dominated by other chromosome in both the values
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1 #if p is dominated by both value then included in n[p]
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1)+1)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2)+1)
    return distance

#Function to carry out the crossover
def crossover_two(parent_1, parent_2, crossover_rate):  # two points crossover
    count = 0
    while(count<10):
     point_1, point_2 = random.sample(range(1, len(parent_1)-1), 2)
     begin = min(point_1, point_2)
     end = max(point_1, point_2)
     child_1 = copy.deepcopy(parent_2[begin:end+1])
     child_2 = copy.deepcopy(parent_1[begin:end+1])
     
     flag=False
     if(random.random() > crossover_rate):
        return parent_1, parent_2
     d1 = dict()
     f = 1
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

#Function to carry out the mutation operator
def mutation(chromosome, expert_size, mutation_rate):
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
    # print("mutation:",new_chromosome)
    return new_chromosome

#Main program starts here
max_gen = 100

#Initialization
pop_size = 30
expert_size=100
with open('SOCIOMETRIC_MATRIX.csv',encoding='utf-8-sig')as csvfile:
      reader=csv.reader(csvfile)
      sociomat=[]
      for row in reader:
       # print(row)
       sociomat.append(row)
for i in range(100):
    for j in range(100):
        sociomat[i][j]=int(sociomat[i][j])
req_mat, all_project_size, chomosome_size = requirement_matrix_gen()
solution=initialization(expert_size,pop_size, chomosome_size)
with open('EFFICIENCY_MATRIX(1).csv',encoding='utf-8-sig')as csvfile:
    Arrayskill_result=csv.reader(csvfile)
    skill_mat=[]
    for row in Arrayskill_result:
      # print(row)
      skill_mat.append(row)
for i in range(100):
    for j in range(8):
        skill_mat[i][j]=int(skill_mat[i][j])
print(skill_mat)
mutation_rate=1
crossover_rate=1
gen_no=0

each_gen_fitness1=[]
each_gen_fitness2=[]
each_gen_fitness1_avg=[]
each_gen_fitness2_avg=[]

start = time.time()
while(gen_no<max_gen):
    function1_values = [function1(solution[i],sociomat,all_project_size) for i in range(0,pop_size)]
    function2_values = [function2(req_mat,skill_mat,all_project_size,solution[i]) for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("Function1:",function1_values)
    print("Function2:",function2_values)
    print("The best front for Generation number ",gen_no, " is")
    front_new=[]
    for values in non_dominated_sorted_solution[0]:
        print(solution[values],end=" ")
        front_new.append(solution[values])
    print("\n")
    
    front_function1_value=[function1(front_new[i],sociomat,all_project_size) for i in range(0,len(front_new))]
    front_function2_value = [function2(req_mat,skill_mat,all_project_size,front_new[i]) for i in range(0,len(front_new))]
    front_function1_value.sort(reverse=True)
    front_function2_value.sort(reverse=True)

    each_gen_fitness1.append(front_function1_value[0])
    each_gen_fitness2.append(front_function2_value[0])

    each_gen_fitness1_avg.append(mean(function1_values))
    each_gen_fitness2_avg.append(mean(function2_values))

    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = copy.deepcopy(solution[:])
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        child_1,child_2=crossover_two(solution[a1],solution[b1],crossover_rate)
        # print("\n1-Child1-child2",child_1,child_2)
        # if(child_1!=solution[a1] and child_1!=solution[b1]):
        #  child1=mutation(child_1,expert_size,mutation_rate)
        #  solution2.append(child_1)
        # else:
        child_1=mutation(child_1,expert_size,mutation_rate)
        # print("\n2-Child1-child2",child_1,child_2)
        solution2.append(child_1)
        
    function1_values2 = [function1(solution2[i],sociomat,all_project_size) for i in range(0,2*pop_size)]
    function2_values2 = [function2(req_mat,skill_mat,all_project_size,solution2[i]) for i in range(0,2*pop_size)]
    print("2-Function1:",function1_values2)
    print("2-Function2:",function2_values2)
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = copy.deepcopy([solution2[i] for i in new_solution])
    gen_no = gen_no + 1
end=time.time()
#Lets plot the final front now




print("Front new ",front_new)
front_function1_values = [function1(front_new[i],sociomat,all_project_size) for i in range(0,len(front_new))]
front_function2_values = [function2(req_mat,skill_mat,all_project_size,front_new[i]) for i in range(0,len(front_new))]
function1 = [i   for i in front_function1_values]
function2 = [j  for j in front_function2_values]
print("Function1-",function1)
print("Function2-",function2)

print("Total Time Taken:",end-start)

x = [i for i in range(1, max_gen+1)]
plt.plot(x, each_gen_fitness1)
plt.xlabel('Genetration number')
# naming the y axis
plt.ylabel('Fitness Values of Function1')
# giving a title to my graph
plt.title('Generation Number Vs Fitness Function1 ')
# function to show the plot
plt.show()

plt.plot(x, each_gen_fitness2)
plt.xlabel('Genetration number')
# naming the y axis
plt.ylabel('Fitness Values of Function2')
# giving a title to my graph
plt.title('Generation Number Vs Fitness Function2 ')
# function to show the plot
plt.show()

plt.plot(x, each_gen_fitness1_avg)
plt.xlabel('Genetration number')
# naming the y axis
plt.ylabel('Avgerage Fitness Values of Function1')
# giving a title to my graph
plt.title('Generation Number Vs Average(Fitness Function1) ')
# function to show the plot
plt.show()

plt.plot(x, each_gen_fitness2_avg)
plt.xlabel('Genetration number')
# naming the y axis
plt.ylabel('Avgerage Fitness Values of Function2')
# giving a title to my graph
plt.title('Generation Number Vs Average(Fitness Function2) ')
# function to show the plot
plt.show()


plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2,c='blue')
plt.grid(True)
plt.show()



