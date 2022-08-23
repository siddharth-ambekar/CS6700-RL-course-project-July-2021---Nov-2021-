from config import *
import time
import numpy as np
from scipy.special import softmax
"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""


class Agent:
    def __init__(self,env):
        self.env_name = env
        self.config = config[self.env_name]
        self.Q_array={};
        self.actions=[];
        self.last_state=None;
        self.last_action=None;
        self.counter={};
        self.state_space={};
        self.beta=0;
        self.incr=0;
        self.weights=None;
        self.grad=None;
        self.count=0;
        self.traj=[];
        self.traj_len=0;
        self.weights=[];
        if self.env_name=='acrobot':
            '''
            Recall that state description is as follows: 
            [cos(theta1),sin(theta1),cos(theta2),sin(theta2),omega1,omega2].
            Actions are:
            [-1:Negative torque, 0:No torque, 1:Positive torque]
            '''
            self.weights=np.random.normal(0,1/(6**0.5),(3,6));
            self.grad=np.zeros((3,6));
            self.beta=0.06;
            self.actions=[-1,0,1];

        elif self.env_name=='taxi':
            '''
            State description:
            [taxi_row,taxi_col,passenger_loc,destination].
            Actions are:
            [0:South, 1:North, 2:East, 3:West, 4:pick-up passenger, 5:drop-off passenger]
            '''
            self.beta=0.0009;
            self.incr=0.09;
            self.actions=[i for i in range(6)];

        elif self.env_name=='kbca':
            self.stage_count=1
            self.beta=0.000001
            self.incr=1
            self.actions=[0,1]

        elif self.env_name=='kbcb':
            self.stage_count=1
            self.beta=0.000001
            self.incr=1
            self.actions=[0,1]

        elif self.env_name=='kbcc':
            self.stage_count=1
            self.beta=0.0001
            self.incr=1
            self.actions=[0,1,2]
            
    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
		
        #return 1
        #raise NotImplementedError
        
        #Here, we shall implement Q-Learning algorithm here
        #Q(s,a)_(t+1) = (1-b_t)*Q(s,a)_t + b_t*(r_t + alpha*max_(b)[ Q(s',b)_t ]);
        #The second term has been obtained from a single step in the episode. We shall first implement an online algorithm.
        
        #We will implement if elif else block;
        
        if self.env_name=='acrobot':
            
            s=np.array([i for i in obs]);
            s=np.reshape(s,(6,1));
                    
            self.weights-=self.beta*self.grad;
            self.grad=np.zeros((3,6));

            A=np.matmul(self.weights,s)
            A=softmax(A);

            n=np.random.choice([0,1,2]);            

            action=self.actions[n]

            self.last_state=s;
            self.last_action=[A,n];
            
        elif self.env_name=='taxi':

            s=obs;
            for i in self.actions:
                if self.Q_array.get((s,i),None)==None:
                    self.Q_array.update({(s,i):np.random.normal(0,5)});
                if self.counter.get((s,i),None)==None:
                    self.counter.update({(s,i):1});

            arr=[];
            for i in self.actions:
                v=np.exp(self.beta*self.Q_array[(s,i)]);
                arr.append(v);
            if len(arr)>0:
                SUM=sum(arr);
                arr=[i/SUM for i in arr];
                action=np.random.choice([i for i in self.actions],p=arr);
            else:
                action=np.random.choice(self.actions);

            if self.counter.get((s,action),None)==None:
                self.counter.update({(s,i):1});
            else:
                self.counter[(s,action)]+=self.incr;

            self.last_action=action;
            self.last_state=s;

        elif self.env_name=='kbca':
            s=(1,0)
        #for i in self.actions:
            #if self.Q_array.get((s,i),None)==None:
                #self.Q_array.update({(s,i):0.0});
            #if self.counter.get((s,i),None)==None:
                #self.counter.update({(s,i):1});
            action=1
            #if self.counter.get((s,action),None)==None:
                #self.counter.update({(s,i):1});
            #else:
                #self.counter[(s,action)]+=self.incr;

            self.last_action=action;
            self.last_state=s;
            self.stage_count=1

        elif self.env_name=='kbcb':
            s=(1,0)
        #for i in self.actions:
            #if self.Q_array.get((s,i),None)==None:
                #self.Q_array.update({(s,i):0.0});
            #if self.counter.get((s,i),None)==None:
                #self.counter.update({(s,i):1});
            action=1
            #if self.counter.get((s,action),None)==None:
                #self.counter.update({(s,i):1});
            #else:
                #self.counter[(s,action)]+=self.incr;

            self.last_action=action;
            self.last_state=s;
            self.stage_count=1
        
        elif self.env_name=='kbcc':
            s=(1,0)
            for i in self.actions:
                if self.Q_array.get((s,i),None)==None:
                    self.Q_array.update({(s,i):0.0});
                if self.counter.get((s,i),None)==None:
                    self.counter.update({(s,i):1});

            arr=[];
            for i in self.actions:
                v=np.exp(self.beta*self.Q_array[(s,i)]);
                arr.append(v);
            if len(arr)>0:
                SUM=sum(arr);
                arr=[i/SUM for i in arr];
                action=np.random.choice([i for i in self.actions],p=arr);
            else:
                action=np.random.choice(self.actions);

            if self.counter.get((s,action),None)==None:
                self.counter.update({(s,i):1});
            #else:
                #self.counter[(s,action)]+=self.incr;

            self.last_action=action;
            self.last_state=s;
            self.stage_count=1

        else:
            action=0;
        return action

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name=='acrobot':
            
            s=np.array([i for i in obs]);
            s=np.reshape(s,(6,1));
            
            [A,n]=self.last_action;
            state=self.last_state;

            A=-A;
            A[n]+=1;

            self.grad+=np.matmul((reward*A).reshape(3,1),s.reshape(1,6));

            A=np.matmul(self.weights,s);
            A=softmax(A);

            n=np.random.choice([0,1,2],p=A.flatten());

            action=self.actions[n];

            self.last_state=s;
            self.last_action=[A,n];

        elif self.env_name=='taxi':

            s=obs;

            beta=self.counter[(self.last_state,self.last_action)];

            for i in self.actions:
                if self.Q_array.get((s,i),None)==None:
                    self.Q_array.update({(s,i):np.random.normal(0,5)});
                if self.counter.get((s,i),None)==None:
                    self.counter.update({(s,i):1});
            arr=[self.Q_array[(s,i)] for i in self.actions];
            m=max(arr);

            self.Q_array[(self.last_state,self.last_action)]=(1-1/beta)*self.Q_array[(self.last_state,self.last_action)]+(1/beta)*(reward+m);

            self.counter[(self.last_state,self.last_action)]+=self.incr;

            arr=[];
            for i in self.actions:
                v=np.exp(self.beta*self.Q_array[(s,i)]);
                arr.append(v);
            if len(arr)>0:
                SUM=sum(arr);
                arr=[i/SUM for i in arr];
                action=np.random.choice([i for i in self.actions],p=arr);
            else:
                action=np.random.choice(self.actions);

            '''
            arr=[[self.Q_array[(s,i)],i] for i in self.actions];
            m=max(arr,key=lambda x: x[0]);

            action=m[1];
            '''
            
            self.last_state=s;
            self.last_action=action;

        elif self.env_name=='kbca':
            s=(obs[self.stage_count-1],self.stage_count)
            if self.stage_count>1:
                beta=self.counter[(self.last_state,self.last_action)];
                ss=1/(beta+1)
                for i in self.actions:
                    if self.Q_array.get((s,i),None)==None:
                        self.Q_array.update({(s,i):0.0});
                    if self.counter.get((s,i),None)==None:
                        self.counter.update({(s,i):1});
                arr=[self.Q_array[(s,i)] for i in self.actions];
                m=max(arr);
                self.Q_array[(self.last_state,self.last_action)]=(1-ss)*self.Q_array[(self.last_state,self.last_action)]+(ss)*(reward+m);
                self.counter[(self.last_state,self.last_action)]+=self.incr;

            if not done:
                arr=[];
                for i in self.actions:
                    if self.Q_array.get((s,i),None)==None:
                        self.Q_array.update({(s,i):0.0});
                    if self.counter.get((s,i),None)==None:
                        self.counter.update({(s,i):1});
                    v=np.exp(self.beta*self.Q_array[(s,i)]);
                    arr.append(v);
                if len(arr)>0:
                    SUM=sum(arr);
                    arr=[i/SUM for i in arr];
                    action=np.random.choice([i for i in self.actions],p=arr);
                else:
                    action=np.random.choice(self.actions);
                        
                self.stage_count+=1  

                '''
                arr=[[self.Q_array[(s,i)],i] for i in self.actions];
                m=max(arr,key=lambda x: x[0]);

                action=m[1];
                '''
                
                self.last_state=s;
                self.last_action=action;
            
            else:
                action=0

        elif self.env_name=='kbcb':
            s=(obs[self.stage_count-1],self.stage_count)
            if self.stage_count>1:
                beta=self.counter[(self.last_state,self.last_action)];
                ss=1/(beta+1)
                for i in self.actions:
                    if self.Q_array.get((s,i),None)==None:
                        self.Q_array.update({(s,i):0.0});
                    if self.counter.get((s,i),None)==None:
                        self.counter.update({(s,i):1});
                arr=[self.Q_array[(s,i)] for i in self.actions];
                m=max(arr);
                self.Q_array[(self.last_state,self.last_action)]=(1-ss)*self.Q_array[(self.last_state,self.last_action)]+(ss)*(reward+m);
                self.counter[(self.last_state,self.last_action)]+=self.incr;

            if not done:
                arr=[];
                for i in self.actions:
                    if self.Q_array.get((s,i),None)==None:
                        self.Q_array.update({(s,i):0.0});
                    if self.counter.get((s,i),None)==None:
                        self.counter.update({(s,i):1});
                    v=np.exp(self.beta*self.Q_array[(s,i)]);
                    arr.append(v);
                if len(arr)>0:
                    SUM=sum(arr);
                    arr=[i/SUM for i in arr];
                    action=np.random.choice([i for i in self.actions],p=arr);
                else:
                    action=np.random.choice(self.actions);
                        
                self.stage_count+=1  

                '''
                arr=[[self.Q_array[(s,i)],i] for i in self.actions];
                m=max(arr,key=lambda x: x[0]);

                action=m[1];
                '''
                
                self.last_state=s;
                self.last_action=action;
            
            else:
                action=0

        elif self.env_name=='kbcc':
            s=(obs[self.stage_count-1],self.stage_count)
            #if self.stage_count>1:
            beta=self.counter[(self.last_state,self.last_action)];
            ss=1/(beta+1)
            for i in self.actions:
                if self.Q_array.get((s,i),None)==None:
                    self.Q_array.update({(s,i):0.0});
                if self.counter.get((s,i),None)==None:
                    self.counter.update({(s,i):1});
            arr=[self.Q_array[(s,i)] for i in self.actions];
            m=max(arr);
            self.Q_array[(self.last_state,self.last_action)]=(1-ss)*self.Q_array[(self.last_state,self.last_action)]+(ss)*(reward+m);
            self.counter[(self.last_state,self.last_action)]+=self.incr;

            if not done:
                arr=[];
                for i in self.actions:
                    if self.Q_array.get((s,i),None)==None:
                        self.Q_array.update({(s,i):0.0});
                    if self.counter.get((s,i),None)==None:
                        self.counter.update({(s,i):1});
                    v=np.exp(self.beta*self.Q_array[(s,i)]);
                    arr.append(v);
                if len(arr)>0:
                    SUM=sum(arr);
                    arr=[i/SUM for i in arr];
                    action=np.random.choice([i for i in self.actions],p=arr);
                else:
                    action=np.random.choice(self.actions);
                        
                self.stage_count+=1  

                '''
                arr=[[self.Q_array[(s,i)],i] for i in self.actions];
                m=max(arr,key=lambda x: x[0]);

                action=m[1];
                '''
                
                self.last_state=s;
                self.last_action=action;
            
            else:
                action=0

        else:
            action=0;

        return action

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        
        if self.env_name=='acrobot':
            
            s=np.array([i for i in obs]);
            s=np.reshape(s,(6,1));
            
            A=np.matmul(self.weights,s);
            A=softmax(A);

            action=np.random.choice(self.actions,p=A.flatten());

        elif self.env_name=='taxi':

            s=obs;
            arr=[];
            for i in self.actions:
                if self.Q_array.get((s,i),None)!=None:
                    arr.append([self.Q_array[(s,i)],i]);
                else:
                    arr.append([-10000,0]);
            m=max(arr,key=lambda x: x[0]);
            action=m[1];
            
        elif self.env_name=='kbca':

            self.stage_count=1
            action=1

        elif self.env_name=='kbcb':

            self.stage_count=1
            action=1

        elif self.env_name=='kbcc':
            self.stage_count=1
            s=(1,0);
            arr=[];
            for i in self.actions:
                if self.Q_array.get((s,i),None)!=None:
                    arr.append([self.Q_array[(s,i)],i]);
                else:
                    arr.append([-10000,0]);
            m=max(arr,key=lambda x: x[0]);
            action=m[1];
            
        else:
            action=0;

        return action

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        if self.env_name=='acrobot':
            
            s=np.array([i for i in obs]);
            s=np.reshape(s,(6,1))
            
            s=np.array([i for i in obs]);
            s=np.reshape(s,(6,1));
            
            A=np.matmul(self.weights,s);
            A=softmax(A);

            action=np.random.choice(self.actions,p=A.flatten());

        elif self.env_name=='taxi':
            
            s=obs;
            arr=[];
            for i in self.actions:
                if self.Q_array.get((s,i),None)!=None:
                    arr.append([self.Q_array[(s,i)],i]);
                else:
                    arr.append([-10000,0]);
            m=max(arr,key=lambda x: x[0]);
            action=m[1];

        elif self.env_name=='kbca':
            s=(obs[self.stage_count-1],self.stage_count)
            if not done:
                arr=[];
                for i in self.actions:
                    if self.Q_array.get((s,i),None)!=None:
                        arr.append([self.Q_array[(s,i)],i]);
                    else:
                        arr.append([-10000,0]);
                m=max(arr,key=lambda x: x[0]);
                action=m[1];
                self.stage_count+=1
            else:
                action=0

        elif self.env_name=='kbcb':
            s=(obs[self.stage_count-1],self.stage_count)
            if not done:
                arr=[];
                for i in self.actions:
                    if self.Q_array.get((s,i),None)!=None:
                        arr.append([self.Q_array[(s,i)],i]);
                    else:
                        arr.append([-10000,0]);
                m=max(arr,key=lambda x: x[0]);
                action=m[1];
                self.stage_count+=1
            else:
                action=0

        elif self.env_name=='kbcc':
            s=(obs[self.stage_count-1],self.stage_count)
            if not done:
                arr=[];
                for i in self.actions:
                    if self.Q_array.get((s,i),None)!=None:
                        arr.append([self.Q_array[(s,i)],i]);
                    else:
                        arr.append([-10000,0]);
                m=max(arr,key=lambda x: x[0]);
                action=m[1];
                self.stage_count+=1
            else:
                action=0

        else:
            action=0;
            
        return action
