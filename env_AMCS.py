import numpy as np

# np.random.seed(2022)

class airport:
    '''
    FH: flight_hours
    FC: flight_cycles
    CD: calendar_days
    '''

    def __init__(self, num_aircraft=2, num_hanger=2, max_FH=100, max_FC=100, max_CD=100):
        self.num_aircraft = num_aircraft
        self.num_hanger = num_hanger
        self.max_FH = max_FH  # max FH for each flight
        self.max_FC = max_FC  # max FC for each flight
        self.max_CD = max_CD  # max CD for each flight
        self.check_FH_list = np.zeros((num_aircraft))
        self.check_FC_list = np.zeros((num_aircraft))
        self.check_CD_list = np.zeros((num_aircraft))
        self.hange_list = np.zeros((num_hanger))
        self.FH = np.zeros((num_aircraft))
        self.FC = np.zeros((num_aircraft))
        self.CD = np.zeros((num_aircraft))

    # Check FH of each aircraft
    def check_FH(self):
        for aircraft_i in range(self.num_aircraft):
            self.FH[aircraft_i] = np.random.randint(low=0, high=self.max_FH*1.5)  # Get the FH of aircraft i
            if self.FH[aircraft_i] >= self.max_FH:
                self.check_FH_list[aircraft_i] += 1
        return self.check_FH_list

    # Check FC of each aircraft
    def check_FC(self):
        for aircraft_i in range(self.num_aircraft):
            self.FC[aircraft_i] = np.random.randint(low=0, high=self.max_FC*1.5)  # Get the FC of aircraft i
            if self.FC[aircraft_i] >= self.max_FC:
                self.check_FC_list[aircraft_i] += 1
        return self.check_FC_list

    # Check CD of each aircraft
    def check_CD(self):
        for aircraft_i in range(self.num_aircraft):
            self.CD[aircraft_i] = np.random.randint(low=0, high=self.max_CD*1.5)  # Get the CD of aircraft i
            if self.CD[aircraft_i] >= self.max_CD:
                self.check_CD_list[aircraft_i] += 1
        return self.check_CD_list


    def airport_state(self):
        state_FH = airport.check_FH(self)
        state_FC = airport.check_FC(self)
        state_CD = airport.check_CD(self)
        final_state = state_FH + state_FC + state_CD
        if final_state[0] == 0 and final_state[1] == 0:
            state = 0
        elif final_state[0] == 0 and final_state[1] != 0:
            state = 1
        elif final_state[0] != 0 and final_state[1] == 0:
            state = 2
        else:
            state = 3
        
        return state

    def step(self, a):
        s = airport.airport_state(self)
        if a == 0:  # 00
            if s == 0:
                r = 1
            elif s == 1:
                r = 0
            elif s == 2:
                r = 0
            else:
                r = 0
        if a == 1:  # 01
            if s == 0:
                r = 0
            elif s == 1:
                r = 1
            elif s == 2:
                r = 0
            else:
                r = 0
        if a == 2:  # 01
            if s == 0:
                r = 0
            elif s == 1:
                r = 0
            elif s == 2:
                r = 1
            else:
                r = 0
        if a == 3:  # 01
            if s == 0:
                r = 0
            elif s == 1:
                r = 0
            elif s == 2:
                r = 0
            else:
                r = 1

        # if r == 1:
        #     done = True
        # else:
        #     done = False

        done = True
        s_ = s
        return s_, r, done


if __name__ == '__main__':
    env = airport()
    integrated_state = env.airport_state()  # 0: no abrasion; 1: light check; 2: heavier check; 3: heaviest check
    print(integrated_state)
