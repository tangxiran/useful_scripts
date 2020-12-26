# encoding=utf-8
import numpy as np
import copy
def fun(x):
    dim,dim_1 =x.shape
    result =0
    for i in range(dim):
        result = result + x[i,0]**2
    return result

def get_min_fitness_index_min_fitness(list_to_compute):
    # return best_particle's position and it fitness flaot
    index  = 0
    min_fitness= list_to_compute[index]
    for i in range(len(list_to_compute)):
        # 计算最小值的粒子是哪一个
        if list_to_compute[i]<list_to_compute[index]:
            index=i
            min_fitness=list_to_compute[i]

    return index , min_fitness


class PSO():
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func, dim=80, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.now_iter_time = 0
        self.record_mode =False # 不做记录
        self.w = w  # inertia
        self.c1, self.c2 = c1, c2  # parameters to control personal best, global best respectively
        self.has_constraints = not (lb is None and ub is None) # 无限制全空间搜索

        self.each_particles = [] # all particles init
        self.fitness_compute = func
        self.lb = -1.0  if lb is None else np.array(lb)  # 下限，默认+-1
        self.ub = 1.0   if ub is None else np.array(ub)  # 上限
        for i in range(pop):
            # init all particles,each one is (dim,1),all particles is (pop , (dim,1))
            self.each_particles.append(np.reshape( np.random.uniform( low=self.lb, high=self.ub, size=(  self.dim))
                                                  ,(self.dim,1))
            )
        # 初始化每个粒子的最佳位置记录，最初的是原位置
        self.each_best_place = copy.deepcopy(self.each_particles)
        # print(id(self.each_best_place))
        # print(id(self.all_particles))

        self.each_v = [] # 每个速度
        for i in range(pop):
            # init each v 初始化每个粒子的速度,随机生成
            self.each_v.append(np.reshape(np.random.uniform( low=self.lb, high=self.ub, size=( self.dim)),(self.dim,1)))

        self.each_fitness = [] # each particles fitness
        for i in range(pop):
            # return [pop ,1]
            # 计算初始化的适合度
            (self.each_fitness).append(self.fitness_compute(self.each_particles[i])) # 每个粒子的当前适合度
        self.each_best_fitness = copy.deepcopy(self.each_fitness) # 每个粒子的历史最佳适合度

        # global best  particle ,return index is the position,
        # and the best fitness is how much
        gbest_x_index ,self.gbest_y = get_min_fitness_index_min_fitness(self.each_fitness)
        self.gbest_x = copy.deepcopy(self.each_particles[gbest_x_index]) # dim,1
        self.gbest_y = self.gbest_y # a float number
        self.gbest_y_hist = []  # gbest_y of every iteration 记录所有的最佳变换值
        self.gbest_y_hist.append(self.gbest_y) # gbest_y of every iteration 记录所有的最佳变换值

        # self.X = np.random.uniform( low=self.lb, high=self.ub, size=( self.pop ,  self.dim ) )
        # v_high = self.ub - self.lb
        # self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles，随机初始化一个
        print('finish init')


    def update_V(self):

        for i in range(self.pop):
            r1 = np.random.random(80)  # 每隔维度上的随机值
            r2 = np.random.random(80)  # 每隔维度上的随机值
            # 对每一个粒子进行更新，按照全局最优和个体的历史最优变化位置
            self.each_v[i] = self.w * self.each_v[i]+r1*self.c1*(self.each_best_place[i]-self.each_particles[i])\
                          +r2*self.c2*(self.gbest_x-self.each_particles[i])



    def update_X(self):
        for i in range(self.pop):
            self.each_particles[i] = self.each_particles[i] + self.each_v[i]

        if self.has_constraints:
            for i in range(self.pop):
                # 限定自变量取值为lb，ub，强制转换

                self.each_particles[i]=np.clip(self.each_particles[i],self.lb,self.ub)

    def cal_fitness(self):
        # calculate y for every x in X
        for i in range(self.pop):
            self.each_fitness[i]= self.fitness_compute(self.each_particles[i])
        return self.each_fitness

    def update_pbest(self):
        '''
        personal best,update
        change the person best position ,person best value
        :return:
        '''
        # self.each_best_placeis change if now the fitness is better
        # best fitness is recorded
        for i in range(self.pop):
            if self.each_fitness[i]<self.each_best_fitness[i]:
                self.each_best_fitness[i] = self.each_fitness[i]
                self.each_best_place[i] = self.each_particles[i]

    def update_gbest(self):
        '''
        global best
        :return: 全局的最优值，x为全局最优位置
        '''
        for i in range(self.pop):
            if self.each_best_fitness[i]<self.gbest_y:
                self.gbest_x = copy.deepcopy(self.each_best_place[i])
                self.gbest_y = copy.deepcopy(self.each_best_fitness[i])



    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.now_iter_time = self.now_iter_time+1
            print('now is ',str(self.now_iter_time),' iteration')
            # all iters  in the
            self.update_V()
            # self.recorder()
            self.update_X()
            self.cal_fitness()

            self.update_pbest()
            self.update_gbest()

            now_best_error = self.gbest_y
            print('now error is ',now_best_error)
            self.gbest_y_hist.append(now_best_error)
        self.best_x, self.best_y = self.gbest_x, now_best_error
        return self.best_x, self.best_y

    fit = run

if __name__ == '__main__':
    # 自己设置的结果，如何改进？
    pso =  PSO(func=fun, dim=80, pop=80*2*2*2, max_iter=1000, lb=None, ub=None, w=0.9, c1=0.5, c2=0.5)
    pso.run()
