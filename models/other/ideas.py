class LastLoss:
    def __init__(self, ll = 0):
         self._ll = ll
      
    # getter method
    def get_ll(self):
        return self._ll
      
    # setter method
    def set_ll(self, x):
        self._ll = x

def print_value(ll):
    ll.set_ll(20)
    print(ll)

def main():
    ll = LastLoss()
    print_value(ll)

    print(ll.get_ll())

if __name__ == '__main__':
    main()
    
