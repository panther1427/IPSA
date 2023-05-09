'''
Reflection upon solution:
    valid: 
    We needed to check weather that the queens threatned eachother. This can be done by evaluating 
    the diagonals, rows and columns.

    print_solution: 
    An exercise in formatting. Run through the solution for each row.

    solve:
    We quickly established that we needed a for loop to set the queens. After that, we found that we 
    had to check wether that the position was valid. The struggle came with finding out that if the
    solution was not valid, we had to delete the former position, and continue looping. 
'''
def valid(r1, c1, r2, c2):
    return r1 != r2 and c1 != c2 and abs(r1 - r2) != abs(c1 - c2)


def print_solution(solution):
    print(f"Size of board: {len(solution)}")
    print("Column : ", end = "")
    print(*[i for i in range(len(solution))], sep = "")

    for i in range(len(solution)):
        print(f"Row {i}  : ", end = "")
        print("." * (solution[i]) + "Q" + "." * (len(solution) - (solution[i] + 1)))
    

def solve(solution, n):

    if len(solution) == n:
        #print(f"Done! Returning with {solution}")
        return solution

    if n == 1:
        return [0]
    
    # Gennemgår alle søjler for den givne række (afhængig af len(solution))
    for i in range(n):
        allvalid = True

        if len(solution) > 0:
            #Checks all of the prior solutions
            for j in range(len(solution)):
                if not valid(j, solution[j], len(solution), i):
                    allvalid = False
        
        if allvalid:
            x = solve(solution + [i], n)
            if len(x) == n:
                return x
                  
    return solution[:len(solution)-1]

        
print_solution(solve([], n=6))
