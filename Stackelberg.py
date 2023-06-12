# -*- coding: utf-8 -*-
"""


@author: H
"""
import numpy as np
import math
import nashpy


class Nash():
    
    def __init__(self,ini_state,actions=[0,1,2,3,4],alpha=0.1,Gamma=0.8):
        self.currentstate=ini_state
        self.actions=actions
        self.alpha = alpha
        self.gamma = Gamma
        self.policy = EpsGreedyQPolicy()
        #self.q={}
        #self.q[ini_state]={0}
        self.q, self.q_o = {}, {}
        self.q[ini_state] = {}
        self.q_o[ini_state] = {}
        # pi (my and opponent)
        self.pi, self.pi_o = {}, {}
        
        #初始化动作
        self.pi[ini_state] = np.array([1, 0, 0, 0, 0])
        self.pi_o[ini_state] = np.array([1, 0, 0, 0, 0])
        
       
       
        self.previous_action = None
        self.reward_history = []
        self.pi_history = []
         # nash q value
        self.nashq = {}
        self.nashq[ini_state] = 0
        
    def act(self, currentstate, playerstate, training=True):
        #全都默认training
        if training:
            action_id = self.policy.select_action(self.pi[self.currentstate],currentstate, self.q, self.q_o, playerstate)
            action = self.actions[action_id]
            self.previous_action = action
            print('self.pi[self.currentstate]',self.pi[self.currentstate],playerstate)
        elif training:
            action_id = self.policy.select_greedy_action(self.pi[self.currentstate],currentstate, self.q, self.q_o,playerstate)
            action = self.actions[action_id]
            print('actionQ',self.pi[self.currentstate],playerstate)
            #print(2)

        return action
   
    def actQ(self, currentstate, playerstate, training=True):
        #全都默认training
        if training:
            action_id = self.policy.select_greedy_action(self.pi[self.currentstate],currentstate, self.q, self.q_o,playerstate)
            action = self.actions[action_id]
            print('actionQ',self.pi[self.currentstate],playerstate)
            #print(2)

        return action
   
    def observe(self, currentstate, nextstate,reward,reward_o,action1, action2):
        self.check_new_state(currentstate)
        self.check_new_state(nextstate)
        self.learn(currentstate,nextstate,reward,reward_o,action1, action2)
        print(self.currentstate, self.actions,self.q,2)
        return self.q
        #return pi  
    def check_new_state(self, state):
        if state not in self.q.keys():
            self.q[state] = {}
            self.q_o[state] = {}
            #print("state not in q.keys")
            #print(state,self.q.keys())
        #print(self.q.keys(),5)    
        for action1 in self.actions:
            for action2 in self.actions:
                if state not in self.pi.keys():
                    #self.pi[state] = np.repeat(
                    #    1.0/len(self.actions), len(self.actions))
                   #self.pi[state] = np.array([1, 0, 0, 0, 0])
                    self.pi[state] = {}
                   
                   #self.pi[state] = {}
                    #self.v[state] = np.random.random()
                    #print("state not in pi.keys",pi)
                    #就是给q[状态][联合动作]赋相同的初值
                if (action1, action2) not in self.q[state].keys():
                    self.q[state][(action1, action2)] = 0
                    self.q_o[state][(action1, action2)] = 0
                    #print("action1, action2 not in q[state].keys",self.q)
    def learn(self, currentstate, nextstate, reward, reward_o, action1, action2):
        self.reward_history.append(reward)
        #计算下一状态的均衡
        self.pi[nextstate], self.pi_o[nextstate] = self.compute_pi(nextstate)
        #print(self.pi[nextstate], self.pi_o[nextstate],3144)
        self.nashq[nextstate] = self.compute_nashq(nextstate)
        #compute pi会返回两个值，一个是第一个卫星player的动作选择概率分布（或者说是策略）
        self.pi_history.append(self.pi[state][0])
        self.q[currentstate][(action1, action2)] = self.compute_q(currentstate, nextstate, reward, action1, action2, self.q)
        self.q_o[currentstate][(action1, action2)] = self.compute_q(currentstate, nextstate, reward_o, action1, action2, self.q_o)    
    # 计算对应状态下动作的概率 row_q_1赋值给q_1，row_q_2赋值给q_2,然后求解纳什均衡 智能体需要学习对方智能体策略的原因是求解纳什均衡时要知道双方Q值，而智能体只知道自己的Q值不知道对方的 因此需要学习对方的
    #这里直接根据状态 返回一个纳什均衡点 也是就当前状态下agent1猜测的均衡点 包含agent1和agent2会选的动作 并且返回的是数组pi[state]= pi[0][0] 那么pi[state][0]就表示数组中的第一个元素
    def compute_pi(self, state):
        #global Meanmax
        q_1, q_2 = [], []
        for action1 in self.actions:
            row_q_1, row_q_2 = [], []
            for action2 in self.actions:
                joint_action = (action1, action2)
                row_q_1.append(self.q[state][joint_action])
                row_q_2.append(self.q_o[state][joint_action])
            q_1.append(row_q_1)
            q_2.append(row_q_2)
        #print (q_1,q_2)
        game = nashpy.Game(q_1, q_2)
        equilibria = game.support_enumeration()
       # print (q_1,q_2,equilibria)
        pi = []
        purenum=0
        #print (pi) pi是先定义为空然后将均衡点赋值进去
        for eq in equilibria:
           # if eq == 0 or eq == 1: 不应该这样表示，这样表示因为这个动作数以及各种组合相关
             pure=[]
             pure.append(eq)
             if 1 in pure[0][0] and 1 in pure[0][1]:
                 pi.append(eq)
                 purenum=purenum+1
        Qmax=0   
        Meanmax=0
        for k in range(purenum):
            for m in range(5):
                for n in range(5):
                    Q=pi[k][0][m]*pi[k][1][n]*self.q[state][(m, n)]+pi[k][0][m]*pi[k][1][n]*self.q_o[state][(m, n)] 
                    if Q>Qmax:
                        Qmax=Q
                        Meanmax=k
        #print (pi,pi[0][0],pi[0][1],314)
        print("Meanmax",Meanmax)
        return pi[Meanmax][0], pi[Meanmax][1]  
    # nash点处的Q值 
    def compute_nashq(self, state):
        nashq = 0
        for action1 in self.actions:
             for action2 in self.actions:
                nashq += self.pi[state][action1]*self.pi_o[state][action2] *self.q[state][(action1, action2)]
        #print(nashq,12)
        return nashq
    
     # Q值的更新 加入下一状态的参数 previous这里没赋值 因为两次调用该函数 所以第一次的输出是agent0的，第二个输出的是agent1的
    def compute_q(self, currentstate, nextstate, reward, action1, action2, q):
        if (action1, action2) not in q[state].keys():
            q[currentstate][(action1, action2)] = 0.0
        q_old = q[currentstate][(action1, action2)]
        updated_q = q_old + (self.alpha * (reward + self.gamma*self.nashq[nextstate] - q_old))
        updated_q =round(updated_q,2)
        print(currentstate, action1, action2, q_old,updated_q,11)        
        return updated_q 
    
class MatrixGame():
    def __init__(self, action1, action2, currentstate):
        self.reward_matrix = self._create_reward_table(action1,action2,currentstate)
        self.reward_matrixS = self._create_reward_tableS(currentstate)
        #这个地方传参需要单独传，不会用上面传的那个,r收益第一项为第一个player1的收益，第二个为player2的收益，所以应该在self.reward_matrix[action1][action2]后面加[0],或[1]进行提取
    def step(self, action1, action2, currentstate):
        print(action1,action2, currentstate,13)
     
       
        r=self.reward_matrix[action1][action2]
        r1 = self.reward_matrix[action1][action2][0]
        r2 = self.reward_matrix[action1][action2][1]
        Raction=[0,0.05,0.1,0.15,0.2]
        t1=Raction[action1]
        t2=Raction[action2]
        #表示状态转移
        f1=currentstate[0]/math.sqrt(currentstate[0]*(1-currentstate[0]))
        f2=currentstate[1]/math.sqrt(currentstate[1]*(1-currentstate[1]))   
        w=r1*(f1**(1.6))+r2*(f2**(1.6))+(1-currentstate[0]-currentstate[1])*1.6
        g1=r1*(f1**(1.6))/w
        g2=r2*(f2**(1.6))/w
       #下一个状态
        nextstate_0=currentstate[0]+g1*(1-currentstate[0])*0.4+(g1/(g1+g2))*currentstate[1]*0.6*t1*(t1+t2)-currentstate[0]*((1-g1)*0.4+(g2/(g1+g2))*0.6*t2*(t1+t2))
        nextstate_1=currentstate[1]+g2*(1-currentstate[1])*0.4+(g2/(g1+g2))*currentstate[0]*0.6*t2*(t1+t2)-currentstate[1]*((1-g2)*0.4+(g1/(g1+g2))*0.6*t1*(t1+t2))              
        nextstate=(round(nextstate_0,2),round(nextstate_1,2))
        print(currentstate[0],action1,action2,r,r1,r2,g1,g2,t1,t2,nextstate,9)
        return None, r1, r2, nextstate
   
    #需要把状态附上去，修改收益矩阵
    def _create_reward_table(self, action1, action2, currentstate):
        print(action1)
        #r1=((self.state[0]-c*a1)*(self.state[1]+a1)+a1*(self.state[1]-c*a2))/(self.state[0]*self.state[1]+self.state[0]*a1+self.state[1]*a2);
        #r2=((self.state[1]-c*a2)*(self.state[0]+a2)+a2*(self.state[0]-c*a1))/(self.state[0]*self.state[1]+self.state[0]*a1+self.state[1]*a2);

        reward_matrix = [
                                [[1, 1], [currentstate[0]/(currentstate[0]+0.05), 1+(0.025*currentstate[0]-0.00125)/((currentstate[0]+0.05)*currentstate[1])], [currentstate[0]/(currentstate[0]+0.1), 1+(0.05*currentstate[0]-0.005)/((currentstate[0]+0.1)*currentstate[1])], [currentstate[0]/(currentstate[0]+0.15), 1+(0.075*currentstate[0]-0.01125)/((currentstate[0]+0.15)*currentstate[1])], [currentstate[0]/(currentstate[0]+0.2), 1+(0.1*currentstate[0]-0.02)/((currentstate[0]+0.2)*currentstate[1])]],
                                [[1+(0.025*currentstate[1]-0.00125)/((currentstate[1]+0.05)*currentstate[0]), currentstate[1]/(currentstate[1]+0.05)], [1-(0.025*currentstate[1]+0.0025)/((currentstate[1]+0.05)*currentstate[0]+0.05*currentstate[1]), 1-(0.025*currentstate[0]+0.0025)/((currentstate[1]+0.05)*currentstate[0]+0.05*currentstate[1])], [1-(0.075*currentstate[1]+0.00375)/((currentstate[1]+0.05)*currentstate[0]+0.1*currentstate[1]), 1-0.0075/((currentstate[1]+0.05)*currentstate[0]+0.1*currentstate[1])], [1-(0.125*currentstate[1]+0.005)/((currentstate[1]+0.05)*currentstate[0]+0.15*currentstate[1]), 1-(0.025*currentstate[0]+0.015)/((currentstate[1]+0.05)*currentstate[0]+0.15*currentstate[1])], [1-(0.175*currentstate[1]+0.00625)/((currentstate[1]+0.05)*currentstate[0]+0.2*currentstate[1]), 1-(0.05*currentstate[0]+0.025)/((currentstate[1]+0.05)*currentstate[0]+0.2*currentstate[1])]],
                                [[1+(0.05*currentstate[1]-0.005)/((currentstate[1]+0.1)*currentstate[0]), currentstate[1]/(currentstate[1]+0.1)], [1-0.0075/((currentstate[1]+0.1)*currentstate[0]+0.05*currentstate[1]), 1-(0.075*currentstate[0]+0.00375)/((currentstate[1]+0.1)*currentstate[0]+0.05*currentstate[1])], [1-(0.05*currentstate[1]+0.01)/((currentstate[1]+0.1)*currentstate[0]+0.1*currentstate[1]), 1-(0.05*currentstate[0]+0.01)/((currentstate[1]+0.1)*currentstate[0]+0.1*currentstate[1])], [1-(0.1*currentstate[1]+0.0125)/((currentstate[1]+0.1)*currentstate[0]+0.15*currentstate[1]), 1-(0.025*currentstate[0]+0.01875)/((currentstate[1]+0.1)*currentstate[0]+0.15*currentstate[1])], [1-(0.15*currentstate[1]+0.015)/((currentstate[1]+0.1)*currentstate[0]+0.2*currentstate[1]), 1-0.03/((currentstate[1]+0.1)*currentstate[0]+0.2*currentstate[1])]],
                                [[1+(0.075*currentstate[1]-0.01125)/((currentstate[1]+0.15)*currentstate[0]), currentstate[1]/(currentstate[1]+0.15)], [1-(0.035*currentstate[1]+0.015)/((currentstate[1]+0.15)*currentstate[0]+0.05*currentstate[1]), 1-(0.125*currentstate[0]+0.005)/((currentstate[1]+0.15)*currentstate[0]+0.05*currentstate[1])], [1-(0.025*currentstate[1]+0.01875)/((currentstate[1]+0.15)*currentstate[0]+0.1*currentstate[1]), 1-(0.1*currentstate[0]+0.0125)/((currentstate[1]+0.15)*currentstate[0]+0.1*currentstate[1])], [1-(0.075*currentstate[1]+0.0225)/((currentstate[1]+0.15)*currentstate[0]+0.15*currentstate[1]), 1-(0.075*currentstate[0]+0.0225)/((currentstate[1]+0.15)*currentstate[0]+0.15*currentstate[1])], [1-(0.125*currentstate[1]+0.02625)/((currentstate[1]+0.15)*currentstate[0]+0.2*currentstate[1]), 1-(0.05*currentstate[0]+0.035)/((currentstate[1]+0.15)*currentstate[0]+0.2*currentstate[1])]],
                                [[1+(0.1*currentstate[1]-0.02)/((currentstate[1]+0.2)*currentstate[0]), currentstate[1]/(currentstate[1]+0.2)], [1-(0.05*currentstate[1]+0.025)/((currentstate[1]+0.2)*currentstate[0]+0.05*currentstate[1]), 1-(0.175*currentstate[0]+0.00625)/((currentstate[1]+0.2)*currentstate[0]+0.05*currentstate[1])], [1-0.03/((currentstate[1]+0.2)*currentstate[0]+0.1*currentstate[1]), 1-(0.15*currentstate[0]+0.015)/((currentstate[1]+0.2)*currentstate[0]+0.1*currentstate[1])], [1-(0.05*currentstate[1]+0.035)/((currentstate[1]+0.2)*currentstate[0]+0.15*currentstate[1]), 1-(0.125*currentstate[0]+0.02625)/((currentstate[1]+0.2)*currentstate[0]+0.15*currentstate[1])], [1-(0.1*currentstate[1]+0.04)/((currentstate[1]+0.2)*currentstate[0]+0.2*currentstate[1]), 1-(0.1*currentstate[0]+0.04)/((currentstate[1]+0.2)*currentstate[0]+0.2*currentstate[1])]]

                            ]
        #print(reward_matrix[0][0],action1)
        return reward_matrix
    def _create_reward_tableS(self, currentstate):
        
        reward_matrix1 = [  [1, currentstate[0]/(currentstate[0]+0.05),currentstate[0]/(currentstate[0]+0.1),currentstate[0]/(currentstate[0]+0.15),currentstate[0]/(currentstate[0]+0.2)],
                    [1+(0.025*currentstate[1]-0.00125)/((currentstate[1]+0.05)*currentstate[0]),1-(0.025*currentstate[1]+0.0025)/((currentstate[1]+0.05)*currentstate[0]+0.05*currentstate[1]),1-(0.075*currentstate[1]+0.00375)/((currentstate[1]+0.05)*currentstate[0]+0.1*currentstate[1]),1-(0.125*currentstate[1]+0.005)/((currentstate[1]+0.05)*currentstate[0]+0.15*currentstate[1]), 1-(0.175*currentstate[1]+0.00625)/((currentstate[1]+0.05)*currentstate[0]+0.2*currentstate[1])],
                    [1+(0.05*currentstate[1]-0.005)/((currentstate[1]+0.1)*currentstate[0]),1-0.0075/((currentstate[1]+0.1)*currentstate[0]+0.05*currentstate[1]), 1-(0.05*currentstate[1]+0.01)/((currentstate[1]+0.1)*currentstate[0]+0.1*currentstate[1]),1-(0.1*currentstate[1]+0.0125)/((currentstate[1]+0.1)*currentstate[0]+0.15*currentstate[1]),1-(0.15*currentstate[1]+0.015)/((currentstate[1]+0.1)*currentstate[0]+0.2*currentstate[1])],
                    [1+(0.075*currentstate[1]-0.01125)/((currentstate[1]+0.15)*currentstate[0]),1-(0.035*currentstate[1]+0.015)/((currentstate[1]+0.15)*currentstate[0]+0.05*currentstate[1]),1-(0.025*currentstate[1]+0.01875)/((currentstate[1]+0.15)*currentstate[0]+0.1*currentstate[1]),1-(0.075*currentstate[1]+0.0225)/((currentstate[1]+0.15)*currentstate[0]+0.15*currentstate[1]), 1-(0.125*currentstate[1]+0.02625)/((currentstate[1]+0.15)*currentstate[0]+0.2*currentstate[1])],
                    [1+(0.1*currentstate[1]-0.02)/((currentstate[1]+0.2)*currentstate[0]),1-(0.05*currentstate[1]+0.025)/((currentstate[1]+0.2)*currentstate[0]+0.05*currentstate[1]),1-0.03/((currentstate[1]+0.2)*currentstate[0]+0.1*currentstate[1]),1-(0.05*currentstate[1]+0.035)/((currentstate[1]+0.2)*currentstate[0]+0.15*currentstate[1]),1-(0.1*currentstate[1]+0.04)/((currentstate[1]+0.2)*currentstate[0]+0.2*currentstate[1])]

                  ] 
        reward_matrix2 = [
                    [ 1, 1+(0.025*currentstate[0]-0.00125)/((currentstate[0]+0.05)*currentstate[1]), 1+(0.05*currentstate[0]-0.005)/((currentstate[0]+0.1)*currentstate[1]),1+(0.075*currentstate[0]-0.01125)/((currentstate[0]+0.15)*currentstate[1]), 1+(0.1*currentstate[0]-0.02)/((currentstate[0]+0.2)*currentstate[1])],
                    [ currentstate[1]/(currentstate[1]+0.05), 1-(0.025*currentstate[0]+0.0025)/((currentstate[1]+0.05)*currentstate[0]+0.05*currentstate[1]), 1-0.0075/((currentstate[1]+0.05)*currentstate[0]+0.1*currentstate[1]), 1-(0.025*currentstate[0]+0.015)/((currentstate[1]+0.05)*currentstate[0]+0.15*currentstate[1]), 1-(0.05*currentstate[0]+0.025)/((currentstate[1]+0.05)*currentstate[0]+0.2*currentstate[1])],
                    [ currentstate[1]/(currentstate[1]+0.1), 1-(0.075*currentstate[0]+0.00375)/((currentstate[1]+0.1)*currentstate[0]+0.05*currentstate[1]), 1-(0.05*currentstate[0]+0.01)/((currentstate[1]+0.1)*currentstate[0]+0.1*currentstate[1]), 1-(0.025*currentstate[0]+0.01875)/((currentstate[1]+0.1)*currentstate[0]+0.15*currentstate[1]), 1-0.03/((currentstate[1]+0.1)*currentstate[0]+0.2*currentstate[1])],
                    [ currentstate[1]/(currentstate[1]+0.15), 1-(0.125*currentstate[0]+0.005)/((currentstate[1]+0.15)*currentstate[0]+0.05*currentstate[1]), 1-(0.1*currentstate[0]+0.0125)/((currentstate[1]+0.15)*currentstate[0]+0.1*currentstate[1]),1-(0.075*currentstate[0]+0.0225)/((currentstate[1]+0.15)*currentstate[0]+0.15*currentstate[1]), 1-(0.05*currentstate[0]+0.035)/((currentstate[1]+0.15)*currentstate[0]+0.2*currentstate[1])],
                    [ currentstate[1]/(currentstate[1]+0.2), 1-(0.175*currentstate[0]+0.00625)/((currentstate[1]+0.2)*currentstate[0]+0.05*currentstate[1]), 1-(0.15*currentstate[0]+0.015)/((currentstate[1]+0.2)*currentstate[0]+0.1*currentstate[1]), 1-(0.125*currentstate[0]+0.02625)/((currentstate[1]+0.2)*currentstate[0]+0.15*currentstate[1]), 1-(0.1*currentstate[0]+0.04)/((currentstate[1]+0.2)*currentstate[0]+0.2*currentstate[1])]

                         ] 
        game = nashpy.Game(reward_matrix1,reward_matrix2)
        #print("game",game)
        equilibria = game.support_enumeration()       
        pi = []
         #print (pi) pi是先定义为空然后将均衡点赋值进去
        purenum=0                    
        for eq in equilibria:
            #print(eq)
            pure=[]
            pure.append(eq)
            if 1 in pure[0][0] and 1 in pure[0][1]:
                pi.append(eq)
                purenum=purenum+1
        Qmax=0   
        flag1=0
        action1=0
        action2=0
        for k in range(purenum):
            for m in range(0,5):
                for n in range(0,5):
                    Q=pi[k][0][m]*pi[k][1][n]*reward_matrix1[m][n]+pi[k][0][m]*pi[k][1][n]*reward_matrix2[m][n]          
                    if Q>Qmax:
                        Qmax=Q
                        flag1=k
                        #print (reward_matrix1[m][n],reward_matrix2[m][n],Qmax,flag1)
        
        for i in range(0,5):
            for j in range(0,5):
                if pi[flag1][0][i]!=0 and pi[flag1][1][j]!=0:
                    action1=i
                    action2=j
                    break                          
        #print("action",action1,action2)
        r1 = reward_matrix1[action1][action2]
        r2 =  reward_matrix2[action1][action2]
        Raction=[0,0.05,0.1,0.15,0.2]
        t1=Raction[action1]
        t2=Raction[action2]
        #表示状态转移
        f1=currentstate[0]/math.sqrt(currentstate[0]*(1-currentstate[0]))
        f2=currentstate[1]/math.sqrt(currentstate[1]*(1-currentstate[1]))   
        w=r1*(f1**(1.6))+r2*(f2**(1.6))+(1-currentstate[0]-currentstate[1])*1.6
        g1=r1*(f1**(1.6))/w
        g2=r2*(f2**(1.6))/w
        #0.4学习率
        nextstate_0=currentstate[0]+g1*(1-currentstate[0])*0.4+(g1/(g1+g2))*currentstate[1]*0.6*t1*(t1+t2)-currentstate[0]*((1-g1)*0.4+(g2/(g1+g2))*0.6*t2*(t1+t2))
        nextstate_1=currentstate[1]+g2*(1-currentstate[1])*0.4+(g2/(g1+g2))*currentstate[0]*0.6*t2*(t1+t2)-currentstate[1]*((1-g2)*0.4+(g1/(g1+g2))*0.6*t1*(t1+t2))
        nextstate=(round(nextstate_0,2),round(nextstate_1,2))
        return action1,action2,reward_matrix1[action1][action2],reward_matrix2[action1][action2],nextstate
    
class EpsGreedyQPolicy():
    def __init__(self, epsilon=0.6, decay_rate=1):
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def select_action(self, q_values, currentstate, q, q_o, playerstate):
        global playerindex
        #global flag1
        if playerstate==currentstate[0]:
           playerindex=0
        elif playerstate==currentstate[1]:
            playerindex=1
        print('q_values',q_values)
        assert q_values.ndim == 1
       # nb_actions = q_values.shape[0]
        #print(nb_actions,33) 
        global validactions
        global validactions1
        global validactions2
        if playerstate>0.1:
            validactions=5
        elif playerstate>0.075 and playerstate<=0.1:
            validactions=4
        elif playerstate>0.05 and playerstate<=0.075:
            validactions=3
        elif playerstate>0.025 and playerstate<=0.05:
            validactions=2  
        elif playerstate>0 and playerstate<=0.025:
            validactions=1
        #这个地方动作选取需改变 因为双方在选动作时除了随机就是按均衡策略选    
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, validactions)
            #print(action,22222)
            return action
        else:
            #action = np.argmax(q_values)
            if currentstate[0]>0.1:
                validactions1=5
            elif currentstate[0]>0.075 and currentstate[0]<=0.1:
                validactions1=4
            elif currentstate[0]>0.05 and currentstate[0]<=0.075:
                validactions1=3
            elif currentstate[0]>0.025 and currentstate[0]<=0.05:
                validactions1=2  
            elif currentstate[0]>0 and currentstate[0]<=0.025:
                validactions1=1
            if currentstate[1]>0.1:
                validactions2=5
            elif currentstate[1]>0.075 and currentstate[1]<=0.1:
                validactions2=4
            elif currentstate[1]>0.05 and currentstate[1]<=0.075:
                validactions2=3
            elif currentstate[1]>0.025 and currentstate[1]<=0.05:
                validactions2=2  
            elif currentstate[1]>0 and currentstate[1]<=0.025:
                validactions2=1
            q_1, q_2 = [], []
            #因为初始并没有这些状态或者动作对应的Q值，初始赋值为空，因此不能根据纳什均衡策略选择 为了能根据均衡策略赋值 这里将为空的q值赋为0 要看会不会对之后的结果有影响
            #print(validactions1,validactions2,q,1111)
            for i in range(0,validactions1):
                row_q_1, row_q_2 = [], []
                for j in range(0,validactions2):
                    joint_action = (i, j)
                    if currentstate not in q.keys() or joint_action not in q[currentstate].keys():
                        q[currentstate][joint_action]=0
                    if currentstate not in q_o.keys() or joint_action not in q_o[currentstate].keys(): 
                        q_o[currentstate][joint_action]=0
                    #print(q[currentstate][joint_action],q_o[currentstate][joint_action])
                    row_q_1.append(q[currentstate][joint_action])
                    row_q_2.append(q_o[currentstate][joint_action])
                q_1.append(row_q_1)
                q_2.append(row_q_2)
            game = nashpy.Game(q_1, q_2)
            equilibria = game.support_enumeration()
               # print (q_1,q_2,equilibria)
            pi = []
                #print (pi) pi是先定义为空然后将均衡点赋值进去          
            purenum=0    
            zjbl=0                
            for eq in equilibria:
               # if eq == 0 or eq == 1: 不应该这样表示，这样表示因为这个动作数以及各种组合相关
                 pure=[]
                 pure.append(eq)
                 if 1 in pure[0][0] and 1 in pure[0][1]:
                     pi.append(eq)
                     purenum=purenum+1
            Qmax=0   
            flag1=0
            for k in range(purenum):
                for m in range(0,validactions1):
                    for n in range(0,validactions2):
                        Q=pi[k][0][m]*pi[k][1][n]*q[state][(m, n)]+pi[k][0][m]*pi[k][1][n]*q_o[state][(m, n)]
                        
                        if Q>Qmax:
                            Qmax=Q
                            flag1=k
                            print (q[state][(m, n)],q_o[state][(m, n)],Qmax,flag1)
            if playerindex==0:
                for i in range(0,validactions1):
                   if pi[flag1][0][i]!=0:
                       zjbl=i
                       break
            elif playerindex==1:
                for i in range(0,validactions2):
                    if pi[flag1][0][i]!=0:
                       zjbl=i
                       break
           # print(action,22223)
            #return pi[0][0], pi[0][1]
            return zjbl   
     
    def select_greedy_action(self, q_values, currentstate, q, q_o, playerstate):
        print("select_greedy_action")
        assert q_values.ndim == 1
        global playerindex
        #global  Meanmax2
        if playerstate==currentstate[0]:
           playerindex=0
        elif playerstate==currentstate[1]:
            playerindex=1
        print('q_values',q_values)
        global validactions
        global validactions1
        global validactions2
        if currentstate[0]>0.1:
            validactions1=5
        elif currentstate[0]>0.075 and currentstate[0]<=0.1:
            validactions1=4
        elif currentstate[0]>0.05 and currentstate[0]<=0.075:
            validactions1=3
        elif currentstate[0]>0.025 and currentstate[0]<=0.05:
            validactions1=2  
        elif currentstate[0]>0 and currentstate[0]<=0.025:
            validactions1=1
        if currentstate[1]>0.1:
            validactions2=5
        elif currentstate[1]>0.075 and currentstate[1]<=0.1:
            validactions2=4
        elif currentstate[1]>0.05 and currentstate[1]<=0.075:
            validactions2=3
        elif currentstate[1]>0.025 and currentstate[1]<=0.05:
            validactions2=2  
        elif currentstate[1]>0 and currentstate[1]<=0.025:
            validactions2=1
        q_1, q_2 = [], []
        #因为初始并没有这些状态或者动作对应的Q值，初始赋值为空，因此不能根据纳什均衡策略选择 为了能根据均衡策略赋值 这里将为空的q值赋为0 
        #print(validactions1,validactions2,q,1111)
        for i in range(0,validactions1):
            row_q_1, row_q_2 = [], []
            for j in range(0,validactions2):
                joint_action = (i, j)
                if currentstate not in q.keys() or joint_action not in q[currentstate].keys():
                    q[currentstate][joint_action]=0
                if currentstate not in q_o.keys() or joint_action not in q_o[currentstate].keys(): 
                    q_o[currentstate][joint_action]=0
                #print(q[currentstate][joint_action],q_o[currentstate][joint_action])
                row_q_1.append(q[currentstate][joint_action])
                row_q_2.append(q_o[currentstate][joint_action])
            q_1.append(row_q_1)
            q_2.append(row_q_2)
        game = nashpy.Game(q_1, q_2)
        #print(game)
        equilibria = game.support_enumeration()
        pi = []
        purenum=0
        ac=0                    
        for eq in equilibria:
             pure=[]
             pure.append(eq)
             if 1 in pure[0][0] and 1 in pure[0][1]:
                 pi.append(eq)
                 purenum=purenum+1
        #print(purenum)          
        Qmax=0
        Meanmax2=0
        for k in range(purenum):
            for m in range(0,validactions1):
                for n in range(0,validactions2):
                    Q=pi[k][0][m]*pi[k][1][n]*q[state][(m, n)]+pi[k][0][m]*pi[k][1][n]*q_o[state][(m, n)]
                    if Q>Qmax:
                        Qmax=Q
                        Meanmax2=k
                        #print (q[state][(m, n)],q_o[state][(m, n)],Qmax,"Meanmax2",Meanmax2,k)
        #print(Meanmax2)
        
        #return pi[Meanmax][0], pi[Meanmax][1]        
            #action=np.where(pi[0][playerindex]!=0)
        if playerindex==0:
            for i in range(0,validactions1):
                if pi[Meanmax2][0][i]!=0:
                    ac=i
                    break
        elif playerindex==1:
            for i in range(0,validactions2):
                if pi[Meanmax2][0][i]!=0:
                    ac=i
                    break
        return ac    

import matplotlib.pyplot as plt
    
if __name__ == "__main__":
    #nb_episode = 10
    sum1=0
    sum2=0
    sum11=0
    sum12=0
    stateactionhistory=[]
    averageRhistory=[]
    agent1Rhistory=[]
    agent1Phistory=[]
    agent2Rhistory=[]
    agent2Phistory=[]
    
    agent1Dhistory=[]
    
    agent2Dhistory=[]
    
    
    agent1=Nash(ini_state=(0.2,0.3),actions=np.arange(5),alpha=0.1, Gamma=0.8)
    agent2=Nash(ini_state=(0.2,0.3),actions=np.arange(5),alpha=0.1, Gamma=0.8)  
    #agent1._init_(ini_state=(0.4,0.3),actions=np.arange(5))
    #print(ini_state)
    state=(0.2,0.3)
    currentstate_0=state[0]
    currentstate_1=state[1]
    currentstate=(currentstate_0,currentstate_1)
    """
    for episode in range(nb_episode):
        print('episode',episode)
        action1 = agent1.act(currentstate, playerstate=currentstate[0])
        action2 = agent2.act(currentstate, playerstate=currentstate[1])
        game=MatrixGame(action1,action2,currentstate)
        _,r1,r2,nextstate=game.step(action1,action2, currentstate)
        stateactionhistory.append((currentstate,action1,action2,nextstate))
        q1=agent1.observe(currentstate=currentstate,nextstate=nextstate,reward=r1,reward_o=r2,action1 =action1,action2 =action2)
        q2=agent2.observe(currentstate=currentstate,nextstate=nextstate,reward=r2,reward_o=r1,action2 =action2,action1 =action1)
        print(r1,r2,nextstate,q1,3)
        if nextstate[0]<0.5 and nextstate[1]<0.5 and  0<nextstate[0] and 0<nextstate[1]:
            currentstate=nextstate
        else:
            currentstate_0= np.random.uniform(0, 0.5)
            currentstate_1= np.random.uniform(0, 0.5)
            currentstate=(round(currentstate_0,2),round(currentstate_1,2))
            break   
    #print(stateactionhistory)  
    """
 
    q10={}
    t=1
    while True:
        #应该把Q值返回给主函数 主函数可以输出Q值 然后通过以收敛为循环条件不断学习Q  
        action1 = agent1.act(currentstate, playerstate=currentstate[0])
        action2 = agent2.act(currentstate, playerstate=currentstate[1])
        game=MatrixGame(action1,action2,currentstate)
        _,r1,r2,nextstate=game.step(action1,action2, currentstate)
        stateactionhistory.append((currentstate,action1,action2,nextstate))
        q1=agent1.observe(currentstate=currentstate,nextstate=nextstate,reward=r1,reward_o=r2,action1 =action1,action2 =action2)
        q2=agent2.observe(currentstate=currentstate,nextstate=nextstate,reward=r2,reward_o=r1,action2 =action2,action1 =action1)
        print(r1,r2,nextstate,q1,3)
        if q1==q10 and t>10000:
           print(t)
           break
        else:
           t=t+1
           q10=q1                                     
        if nextstate[0]<0.5 and nextstate[1]<0.5 and  0<nextstate[0] and 0<nextstate[1]:
            currentstate=nextstate
        else:
            #currentstate_0= np.random.uniform(0, 0.5)
            #currentstate_1= np.random.uniform(0, 0.5)
            #currentstate=(round(currentstate_0,2),round(currentstate_1,2))
            currentstate=(0.2,0.3)
            #注意状态修改 可能会陷入循环
        if (currentstate[0]<0.05 and currentstate[1]<0.05) or (currentstate[0]+currentstate[1]<=0.1) or currentstate[0]<=0.02 or currentstate[1]<=0.02:
            currentstate=(0.2,0.3)
    print(stateactionhistory)    
    
    #需要写一个函数 就是平均收益函数 初始状态就是（0.2,0.3） 然后按照Q表选动作 然后计算收益（平均收益）收益和除以迭代次数  
    currentstate=(0.2,0.3)
    for i in range(500):
        action1 = agent1.actQ(currentstate, playerstate=currentstate[0])
        action2 = agent2.actQ(currentstate, playerstate=currentstate[1])
        game=MatrixGame(action1,action2,currentstate)
        _,r1,r2,nextstate=game.step(action1,action2, currentstate)
        if currentstate[0]<0.5 and currentstate[1]<0.5 and  0<currentstate[0] and 0<currentstate[1] and r1>=0 and r2>=0:
            sum1=sum1+r1*currentstate[0]
            sum2=sum2+r2*currentstate[1]
            sum11=sum11+0.8**(i)*r1*currentstate[0]
            sum12=sum12+0.8**(i)*r2*currentstate[1]
        else:
            sum1=sum1
            sum2=sum2
            sum11=sum11
            sum12=sum12
        print("average reward",q1,currentstate,action1,action2,nextstate,r1,r2,sum1/(i+1),sum2/(i+1),i+1)
        agent1Rhistory.append(r1)
        agent2Rhistory.append(r2)
        #agent1Phistory.append(sum1/(i+1))
        agent1Phistory.append(sum1/(i+1))
        agent2Phistory.append(sum2/(i+1))
        agent1Dhistory.append(sum11/(i+1))
        agent2Dhistory.append(sum12/(i+1))
        averageRhistory.append((currentstate,action1,action2,nextstate,r1,r2,sum1/(i+1),sum2/(i+1),sum11/(i+1),sum12/(i+1),i+1))
        if nextstate[0]<0.5 and nextstate[1]<0.5 and  0<nextstate[0] and 0<nextstate[1] :
            currentstate=nextstate
        #else:
            #r1=0
            #break 
    print(averageRhistory)
    
    sum1f=0
    sum2f=0
    sum1S=0
    sum2S=0
    sum11f=0
    sum12f=0
    sum11S=0
    sum12S=0
    averageRhistoryf=[]
    agent1Rhistoryf=[]
    agent1Phistoryf=[]
    agent2Rhistoryf=[]
    agent2Phistoryf=[]
    agent1Dhistoryf=[]
    agent2Dhistoryf=[]
    averageRhistoryS=[]
    agent1RhistoryS=[]
    agent1PhistoryS=[]
    agent2RhistoryS=[]
    agent2PhistoryS=[]
    agent1DhistoryS=[]
    agent2DhistoryS=[]
    currentstatef=(0.2,0.3)
    currentstateS=(0.2,0.3)
    for i in range(500):
        action1f = 2
        action2f = 2
        game=MatrixGame(action1f,action2f,currentstatef)
        _,r1f,r2f,nextstatef=game.step(action1f,action2f, currentstatef)
        if currentstatef[0]<0.5 and currentstatef[1]<0.5 and  0<currentstatef[0] and 0<currentstatef[1] and r1f>=0 and r2f>=0:
            sum1f=sum1f+r1f*currentstatef[0]
            sum2f=sum2f+r2f*currentstatef[1]
            sum11f=sum11f+0.8**(i)*r1f*currentstatef[0]
            sum12f=sum12f+0.8**(i)*r2f*currentstatef[1]
        else:
            sum1f=sum1f
            sum2f=sum2f  
            sum11f=sum11f
            sum12f=sum12f    
            #print("average reward",currentstatef,action1f,action2f,r1f,r2f,sum1f/(i+1),sum2f/(i+1),currentstatef[0]*sum1f/(i+1),currentstatef[1]*sum2f/(i+1),i+1)
        agent1Rhistoryf.append(r1f)
        agent2Rhistoryf.append(r2f)
        agent1Dhistoryf.append(sum11f/(i+1))
        agent2Dhistoryf.append(sum12f/(i+1))
        agent1Phistoryf.append(sum1f/(i+1))
        agent2Phistoryf.append(sum2f/(i+1))

        averageRhistoryf.append((currentstatef,action1f,action2f,nextstatef,r1f,r2f,sum1f/(i+1),sum2f/(i+1),sum11f/(i+1),sum12f/(i+1),i+1))
        if nextstatef[0]<0.5 and nextstatef[1]<0.5 and  0<nextstatef[0] and 0<nextstatef[1]:
            currentstatef=nextstatef
        #else:
            #r1f=0
            #r2f=0
            #break
            
        game=MatrixGame(action1f,action2f,currentstateS)
        action1S,action2S,r1S,r2S,nextstateS=game._create_reward_tableS(currentstateS)
        if currentstateS[0]<0.5 and currentstateS[1]<0.5 and  0<currentstateS[0] and 0<currentstateS[1] and r1S>=0 and r2S>=0:
            sum1S=sum1S+r1S*currentstateS[0]
            sum2S=sum2S+r2S*currentstateS[1]
            sum11S=sum11S+0.8**(i)*r1S*currentstateS[0]
            sum12S=sum12S+0.8**(i)*r2S*currentstateS[1]
        else:
            sum1S=sum1S
            sum2S=sum2S
            sum11S=sum11S
            sum12S=sum12S
        #print("average reward",currentstateS,action1S,action2S,r1S,r2S,sum1S/(i+1),sum2S/(i+1),currentstateS[0]*sum1S/(i+1),currentstateS[1]*sum2S/(i+1),i+1)
        agent1RhistoryS.append(r1S)
        agent2RhistoryS.append(r2S)
        agent1PhistoryS.append(sum1S/(i+1))
        agent2PhistoryS.append(sum2S/(i+1))
        agent1DhistoryS.append(sum11S/(i+1))
        agent2DhistoryS.append(sum12S/(i+1))
        #.append(currentstateS[0]*sum1S/(i+1))折扣
        #卫星智能体2收益密度、收益、折扣
        averageRhistoryS.append((currentstateS,action1S,action2S,r1S,r2S,sum1S/(i+1),sum2S/(i+1),sum11S/(i+1),sum12S/(i+1),i+1))
        if nextstateS[0]<0.5 and nextstateS[1]<0.5 and  0<nextstateS[0] and 0<nextstateS[1]:
            currentstateS=nextstateS
        #else:
            #r1S=0  
            #r2S=0
            #break
    print(averageRhistory)
    print(averageRhistoryS)
    print(averageRhistoryf)
  

with open ('agent1Phistory.txt','a')as file0:
    print('%s'%agent1Phistory,file=file0)  
with open ('agent1PhistoryS.txt','a')as file1:
    print('%s'%agent1PhistoryS,file=file1)
with open ('agent1Phistoryf.txt','a')as file2:
    print('%s'%agent1Phistoryf,file=file2)
with open ('agent1Rhistory.txt','a')as file0:
    print('%s'%agent1Rhistory,file=file0)  
with open ('agent1RhistoryS.txt','a')as file1:
    print('%s'%agent1RhistoryS,file=file1)
with open ('agent1Rhistoryf.txt','a')as file2:
    print('%s'%agent1Rhistoryf,file=file2)
with open ('agent2Phistory.txt','a')as file0:
    print('%s'%agent2Phistory,file=file0)  
with open ('agent2PhistoryS.txt','a')as file1:
    print('%s'%agent2PhistoryS,file=file1)
with open ('agent2Phistoryf.txt','a')as file2:
    print('%s'%agent2Phistoryf,file=file2)
with open ('agent2Rhistory.txt','a')as file0:
    print('%s'%agent2Rhistory,file=file0)  
with open ('agent2RhistoryS.txt','a')as file1:
    print('%s'%agent2RhistoryS,file=file1)
with open ('agent2Rhistoryf.txt','a')as file2:
    print('%s'%agent2Rhistoryf,file=file2)
    
with open ('agent1Dhistory.txt','a')as file1:
    print('%s'%agent1Dhistory,file=file1)
with open ('agent2Dhistory.txt','a')as file2:
    print('%s'%agent2Dhistory,file=file2)
with open ('agent1DhistoryS.txt','a')as file1:
    print('%s'%agent1DhistoryS,file=file1)
with open ('agent2DhistoryS.txt','a')as file2:
    print('%s'%agent2DhistoryS,file=file2)
with open ('agent1Dhistoryf.txt','a')as file1:
    print('%s'%agent1Dhistoryf,file=file1)
with open ('agent2Dhistoryf.txt','a')as file2:
    print('%s'%agent2Dhistoryf,file=file2)
#with open ('4.txt','a')as file3:
    #print('%s'%agent2PhistoryS,file=file3) 
#with open ('5.txt','a')as file4:
    #print('%s%d%s%s%s%d'%currentstate%action1%nextstate%agent1Phistory%agent1Rhistory,file=file4)
#with open ('6.txt','a')as file5:
    #print('%s'%currentstateS,file=file5) 
#with open ('7.txt','a')as file6:
   # print('%d'%action1,file=file6) 
#with open ('7.txt','a')as file7:
    #print('%d'%action2,file=file7) 
    

    
    
    #plt.plot(np.arange(len(agent1Rhistory)), agent1Rhistory, label="agent1's average payoff density-p")
    plt.plot(np.arange(len(agent1Phistory)), agent1Phistory, label="agent1's average payoff-p")
    #plt.plot(np.arange(len(agent2Rhistory)), agent2Rhistory, label="agent2's average payoff density-p")
    #plt.plot(np.arange(len(agent2Phistory)), agent2Phistory, label="agent2's average payoff-p")
   # plt.plot(np.arange(len(agent1Rhistoryf)), agent1Rhistoryf, label="agent1's average payoff density-f")
    plt.plot(np.arange(len(agent1Phistoryf)), agent1Phistoryf, label="agent1's average payoff-f")
   # plt.plot(np.arange(len(agent1Rhistoryf)), agent1Rhistoryf, label="agent2's average payoff density-f")
    #plt.plot(np.arange(len(agent1Phistoryf)), agent1Phistoryf, label="agent2's average payoff-f")
    #plt.plot(np.arange(len(agent1RhistoryS)), agent1RhistoryS, label="agent1's average payoff density-m")
    plt.plot(np.arange(len(agent1PhistoryS)), agent1PhistoryS, label="agent1's average payoff-m")    
    #plt.plot(np.arange(len(agent1RhistoryS)), agent1RhistoryS, label="agent2's average payoff density-m")
    #plt.plot(np.arange(len(agent1PhistoryS)), agent1PhistoryS, label="agent2's average payoff-m")  
    plt.xlabel("iteration")
    plt.ylabel("average payoff (0)")
    plt.legend()
    #plt.savefig(r"C:\Users\Administrator\Desktop\Implement-of-algorithm\Fig\Nash-Q.png")
    plt.show()
    
    
    plt.plot(np.arange(len(agent1Rhistory)), agent1Rhistory, label="agent1's average payoff density-p")
    plt.plot(np.arange(len(agent1RhistoryS)), agent1RhistoryS, label="agent1's average payoff density-m")
    plt.plot(np.arange(len(agent1Rhistoryf)), agent1Rhistoryf, label="agent1's average payoff density-f")
    plt.xlabel("iteration")
    plt.ylabel("average payoff density(0)")
    plt.legend()
    plt.show()
    
    
    
    plt.plot(np.arange(len(agent2Phistory)), agent2Phistory, label="agent2's average payoff-p")   
    plt.plot(np.arange(len(agent2Phistoryf)), agent2Phistoryf, label="agent2's average payoff-f")
    plt.plot(np.arange(len(agent2PhistoryS)), agent2PhistoryS, label="agent2's average payoff-m")    
    plt.xlabel("iteration")
    plt.ylabel("average payoff (0)")
    plt.legend()
    plt.show()
     
     
    plt.plot(np.arange(len(agent2Rhistory)), agent2Rhistory, label="agent2's average payoff density-p")
    plt.plot(np.arange(len(agent2RhistoryS)), agent2RhistoryS, label="agent2's average payoff density-m")
    plt.plot(np.arange(len(agent2Rhistoryf)), agent2Rhistoryf, label="agent2's average payoff density-f")
    plt.xlabel("iteration")
    plt.ylabel("average payoff density(0)")
    plt.legend()
    plt.show()
     
     
    plt.plot(np.arange(len(agent1Dhistory)), agent1Dhistory, label="agent1's discounted payoff-p")
   
    plt.plot(np.arange(len(agent1Dhistoryf)), agent1Dhistoryf, label="agent1's discounted payoff-f")
  
    plt.plot(np.arange(len(agent1DhistoryS)), agent1DhistoryS, label="agent1's discounted payoff-m")    
    
    plt.xlabel("iteration")
    plt.ylabel("discounted payoff (0)")
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(len(agent2Dhistory)), agent2Dhistory, label="agent2's discounted payoff-p")
   
    plt.plot(np.arange(len(agent2Dhistoryf)), agent2Dhistoryf, label="agent2's discounted payoff-f")
  
    plt.plot(np.arange(len(agent2DhistoryS)), agent2DhistoryS, label="agent2's discounted payoff-m")    
    
    plt.xlabel("iteration")
    plt.ylabel("discounted payoff (0)")
    plt.legend()
   
    plt.show()
           
