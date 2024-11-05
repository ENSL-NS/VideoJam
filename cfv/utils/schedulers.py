from functools import reduce
from math import gcd
from typing import List


class Server:
    """
    A small representation of a server
    """
    def __init__(self, key, value: any) -> None:
        self.key = key
        self.value = value

    def __repr__(self) -> str:
        return """Server=(
            'key': {},
            'value': {},
        )""".format(self.key, self.value)

class WRRScheduling:
    """
    Weighted Round-Robin Scheduling

    Supposing that there is a server set S = {S0, S1, …, Sn-1};
    W(Si) indicates the weight of Si;
    i indicates the server selected last time, and i is initialized with -1;
    cw is the current weight in scheduling, and cw is initialized with zero; 
    max(S) is the maximum weight of all the servers in S;
    gcd(S) is the greatest common divisor of all server weights in S;

    Sources
    -------
    http://kb.linuxvirtualserver.org/wiki/Weighted_Round-Robin_Scheduling
    """

    def __init__(self) -> None:
        self.S: List[Server] = []
        self.W = {}
        self.n = len(self.S)
        self.i = -1
        self.cw = 0

    def schedule(self, s: Server, weight=1) -> None:
        self.S.append(s)
        self.W[s.key] = weight
        self.n = len(self.S)
        
    def update(self, key, weight):
        self.W[key] = weight

    def get_next(self) -> Server:
        """
        Retrieve next server with the heighest weight.
        """
        # return the server if there is only one.
        if len(self.S) == 1:
            return self.S[0]
        while True:
            self.i = (self.i + 1) % self.n
            if (self.i == 0):
                # Python 3.9
                # self.cw = self.cw - math.gcd(*self.W.values())
                
                # 3.5 <= Python <= 3.8.x 
                self.cw = self.cw - reduce(gcd, self.W.values()) 
                if (self.cw <= 0):
                    self.cw = max(self.W.values())
                    if (self.cw == 0):
                        return None
            if self.W[self.S[self.i].key] >= self.cw:
                return self.S[self.i]
            

if __name__ == "__main__":  
    s = WRRScheduling()
    s.schedule(Server(1, "A"), 4)
    s.schedule(Server(2, "B"), 3)
    s.schedule(Server(3, "C"), 2)
    for c in range(27):
        v = s.get_next()
        if not v:
            break
        print(v.value, end=" ")