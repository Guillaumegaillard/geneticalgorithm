'''

Copyright 2020 Ryan (Mohammad) Solgi, Saullo G. P. Castro

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

###############################################################################
###############################################################################
###############################################################################
import numpy as np
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
from itertools import chain as iter_chain
###############################################################################
###############################################################################
###############################################################################

class geneticalgorithm():

    '''  Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.



    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations

    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters={'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None,\
                                       'multiprocessing_ncpus': 1,\
                                       'multiprocessing_engine': None,\
                                       }):


        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_type <string> - 'bool' if all variables are Boolean;
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)

        @param variable_boundaries <numpy array/None> - Default None; leave it
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        @param variable_type_mixed <numpy array/None> - Default None; leave it
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        @param function_timeout <float> - if the given function does not provide
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective
            @ multiprocessing_ncpus <int> - number of cores to run the
            function in more than 1 CPU
            @ multiprocessing_engine <str> - multiprocessing engine. Native
            multiprocessing module and ray.util.multiprocessing are supported.

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        '''
        self.__name__=geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"

        self.f=function
        #############################################################
        #dimension

        self.dim=int(dimension)

        #############################################################
        # input variable type

        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)

        if variable_type_mixed is None:

            if variable_type=='real':
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)


        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"


            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries


        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':

            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"

            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"


            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two."
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)

        #############################################################
        #Timeout
        self.funtimeout=float(function_timeout)

        #############################################################
        # input algorithm's parameters

        self.param=algorithm_parameters

        self.multiprocessing_ncpus = self.param['multiprocessing_ncpus']
        assert isinstance(self.multiprocessing_ncpus, int), "multiprocessing_ncpus not a valid integer"

        self.multiprocessing_engine = self.param['multiprocessing_engine']

        self.pop_s=int(self.param['population_size'])

        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]"

        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1

        self.prob_mut=self.param['mutation_probability']

        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"


        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"

        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"

        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)

        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                print("GA shrinks #evals to 10000000")
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])

        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"


        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else:
            self.mniwi=int(self.param['max_iteration_without_improv'])


    def chunk_list(self,ll,num_chunks):
        chunk_size=len(ll)//num_chunks +1
        return([ll[chunk_size*j:chunk_size*(j+1)] for j in range(num_chunks-1)]+[ll[chunk_size*(num_chunks-1):]])

    def merge_list_map(self,ll):
        #to choose from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        return(list(iter_chain.from_iterable(ll)))
        # return([item for sublist in ll for item in sublist]) 

    def new_babies(self,Xl):
        return(list(map(self.new_baby,Xl)))
        
    def new_baby(self,rvals):

        r1=rvals[0]
        r2=rvals[1]

        pvar1=self.ef_par[r1,: self.dim].copy()
        pvar2=self.ef_par[r2,: self.dim].copy()

        ch=self.cross(pvar1,pvar2,self.c_type)
        ch1=ch[0].copy()
        ch2=ch[1].copy()

        ch1=self.mut(ch1)
        ch2=self.mutmidle(ch2,pvar1,pvar2)

        v1=self.sim(ch1)
        v2=self.sim(ch2)

        return([np.append(ch1,v1),np.append(ch2,v2)])


        #############################################################
    def run(self, plot=False, initial_idv=None):

        if self.multiprocessing_ncpus > 1:
            if self.multiprocessing_engine is None:
                from multiprocessing import Pool
                print("using Python  multiprocessing")
            elif self.multiprocessing_engine == 'ray':
                from ray.util.multiprocessing import Pool
                print("using ray.util.multiprocessing")
            else:
                raise NotImplementedError('Invalid multiprocessing_engine value')

        #############################################################
        # Initial Population

        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')

        self.lsi=len(self.integers[0])
        self.lsr=len(self.reals[0])


        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo1=np.zeros(self.dim+1)
        solo2=np.zeros(self.dim+1)
        var=np.zeros(self.dim)

        if self.multiprocessing_ncpus > 1:
            pool = Pool(self.multiprocessing_ncpus)
            var_list1 = []
        
        if type(initial_idv).__module__=='numpy':#
            var=initial_idv
            obj=self.sim(initial_idv)
            solo1[self.dim]=obj
            for p in range(0, self.pop_s):
                pop[p]=solo1.copy()

        else:
            for p in range(0,self.pop_s):

                var[self.integers[0]]=np.random.randint(self.var_bound[self.integers[0],0],self.var_bound[self.integers[0],1]+1,
                    size=self.lsi)
                solo1[self.integers[0]]=var[self.integers[0]].copy()

                var[self.reals[0]]=self.var_bound[self.reals[0],0]+np.random.random(size=self.lsr)*(
                    self.var_bound[self.reals[0],1]-self.var_bound[self.reals[0],0])
                solo1[self.reals[0]]=var[self.reals[0]].copy()

                if self.multiprocessing_ncpus > 1:
                    var_list1.append(var)

                else:
                    obj=self.sim(var)
                    solo1[self.dim]=obj
                    pop[p]=solo1.copy()

            if self.multiprocessing_ncpus > 1:
                obj_list1 = self.merge_list_map(pool.map(self.super_sim, self.chunk_list(var_list1,self.multiprocessing_ncpus)))
                for p in range(0, self.pop_s):
                    obj = obj_list1[p]
                    solo1[self.dim]=obj
                    pop[p]=solo1.copy()

        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.test_obj=obj
        self.best_variable=var.copy()
        self.best_function=obj
        ##############################################################

        t=1
        counter=0
        while t<=self.iterate:


            self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]



            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report

            self.report.append(pop[0,self.dim])

            ##############################################################
            # Normalizing objective function

            normobj=np.zeros(self.pop_s)

            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)

            else:
                normobj=pop[:,self.dim].copy()

            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################
            # Calculate probability

            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)

            #############################################################
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)

            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()

            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                ran_cross=np.random.random(size=self.par_s)
                ef_par_list=np.where(ran_cross<=self.prob_cross)
                par_count=np.count_nonzero(ef_par_list)

            self.ef_par=par[ef_par_list].copy()

            #############################################################
            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)

            for k in range(0,self.par_s):
                pop[k]=par[k].copy()

            k_list = np.arange(self.par_s, self.pop_s, 2)
            k_list_odds = np.arange(self.par_s+1, self.pop_s, 2)
            ll=(self.pop_s-self.par_s)//2

            rvals=np.random.randint(0, par_count, size=(ll,2))
            if self.multiprocessing_ncpus > 1:
                # outpool=np.array(list(pool.map(self.new_baby, rvals)))
                outpool=np.array(self.merge_list_map(pool.map(self.new_babies, self.chunk_list(rvals,self.multiprocessing_ncpus))))
            else:
                outpool=np.array(list(map(self.new_baby, rvals)))
            pop[k_list]=outpool[:,0]
            pop[k_list_odds]=outpool[:,1]

        #############################################################
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    self.progress(t,self.iterate,status="GA is running...")
                    time.sleep(2)
                    t+=1
                    self.stop_mniwi=True

        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]

        if pop[0,self.dim]<self.best_function:

            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0,self.dim])




        self.output_dict={'variable': self.best_variable, 'function':\
                          self.best_function}
        show=' '*100
        sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush()
        re=np.array(self.report)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()

        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')
##############################################################################
##############################################################################
    def cross(self,x,y,c_type):

        ofs1=x.copy()
        ofs2=y.copy()


        if c_type=='one_point':
            ran=np.random.randint(0,self.dim)

            ofs1[:ran]=y[:ran].copy()
            ofs2[:ran]=x[:ran].copy()


        if c_type=='two_point':
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)

            ofs1[ran1:ran2]=y[ran1:ran2].copy()
            ofs2[ran1:ran2]=x[ran1:ran2].copy()

        if c_type=='uniform':
            ran_a=np.random.random(size=self.dim)
            ofs1=np.where(ran_a<0.5,y,x)
            ofs2=np.where(ran_a<0.5,x,y)

        return np.array([ofs1,ofs2])
###############################################################################

    def mut(self,x):
        if self.lsi>0:
            ran_a=np.random.random(size=self.lsi)
            ran_mut=np.random.randint(self.var_bound[self.integers[0],0],self.var_bound[self.integers[0],1]+1)#,size=self.lsi)        
            x[self.integers[0]]=np.where(ran_a<self.prob_mut,ran_mut,x[self.integers[0]])

        if self.lsr>0:
            ran_a=np.random.random(size=self.lsr)
            ran_mut=self.var_bound[self.reals[0],0]+np.random.random(size=self.lsr)*(self.var_bound[self.reals[0],1]-self.var_bound[self.reals[0],0])        
            x[self.reals[0]]=np.where(ran_a<self.prob_mut,ran_mut,x[self.reals[0]])

        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        minpp=np.where(p1!=p2,np.where(p1>p2,p2,p1),self.var_bound[:,0])
        maxpp_ints=np.where(p1!=p2,np.where(p1>p2,p1,p2),self.var_bound[:,1]+1)
        
        ran_a=np.random.random(size=self.dim)
        
        if self.lsr>0:
            maxpp_reals=np.where(p1!=p2,maxpp_ints,maxpp_ints-1)
            ran_mut=minpp + np.random.random(size=self.dim)*(maxpp_reals-minpp)
            x[self.reals[0]]=np.where(ran_a[self.reals[0]]<self.prob_mut,ran_mut[self.reals[0]],x[self.reals[0]])

        if self.lsi>0:
            ran_mut_ints=np.zeros(self.dim,dtype='int')
            ran_mut_ints[self.integers[0]]=np.random.randint(minpp[self.integers[0]],maxpp_ints[self.integers[0]])
            x[self.integers[0]]=np.where(ran_a[self.integers[0]]<self.prob_mut,ran_mut_ints[self.integers[0]],x[self.integers[0]])

        return x
###############################################################################
    def evaluate(self):
        return self.f(self.temp)
###############################################################################
    def super_sim(self,Xl):
        return(list(map(self.sim,Xl)))
    def sim(self,X):
        self.temp=X.copy()
        obj=None
        try:
            obj=func_timeout(self.funtimeout,self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        return obj

###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
###############################################################################
###############################################################################



