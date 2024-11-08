from pysdd.sdd import SddManager


def get_sdds_for_sums():
    # fmt: off
    # declare the variables that we need
    manager = SddManager(20, 0)
    (
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, 
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9,
    ) = [manager.literal(i) for i in range(1, 21)]

    # declare the boolean expression for all sums (0-18)
    sum0  = x0 & y0
    sum1  = (x0 & y1) | (x1 & y0)
    sum2  = (x0 & y2) | (x1 & y1) | (x2 & y0)
    sum3  = (x0 & y3) | (x1 & y2) | (x2 & y1) | (x3 & y0)
    sum4  = (x0 & y4) | (x1 & y3) | (x2 & y2) | (x3 & y1) | (x4 & y0)
    sum5  = (x0 & y5) | (x1 & y4) | (x2 & y3) | (x3 & y2) | (x4 & y1) | (x5 & y0)
    sum6  = (x0 & y6) | (x1 & y5) | (x2 & y4) | (x3 & y3) | (x4 & y2) | (x5 & y1) | (x6 & y0)
    sum7  = (x0 & y7) | (x1 & y6) | (x2 & y5) | (x3 & y4) | (x4 & y3) | (x5 & y2) | (x6 & y1) | (x7 & y0)
    sum8  = (x0 & y8) | (x1 & y7) | (x2 & y6) | (x3 & y5) | (x4 & y4) | (x5 & y3) | (x6 & y2) | (x7 & y1) | (x8 & y0)
    sum9  = (x0 & y9) | (x1 & y8) | (x2 & y7) | (x3 & y6) | (x4 & y5) | (x5 & y4) | (x6 & y3) | (x7 & y2) | (x8 & y1) | (x9 & y0)
    sum10 = (x1 & y9) | (x2 & y8) | (x3 & y7) | (x4 & y6) | (x5 & y5) | (x6 & y4) | (x7 & y3) | (x8 & y2) | (x9 & y1)
    sum11 = (x2 & y9) | (x3 & y8) | (x4 & y7) | (x5 & y6) | (x6 & y5) | (x7 & y4) | (x8 & y3) | (x9 & y2)
    sum12 = (x3 & y9) | (x4 & y8) | (x5 & y7) | (x6 & y6) | (x7 & y5) | (x8 & y4) | (x9 & y3)
    sum13 = (x4 & y9) | (x5 & y8) | (x6 & y7) | (x7 & y6) | (x8 & y5) | (x9 & y4)
    sum14 = (x5 & y9) | (x6 & y8) | (x7 & y7) | (x8 & y6) | (x9 & y5)
    sum15 = (x6 & y9) | (x7 & y8) | (x8 & y7) | (x9 & y6)
    sum16 = (x7 & y9) | (x8 & y8) | (x9 & y7)
    sum17 = (x8 & y9) | (x9 & y8)
    sum18 = x9 & y9

    # these contraints encapsulate that the different values for each digit are mutually exclusive
    # i.e. the digit *must* take a value (0-9) and if any value is true then no other value can be true
    constraints_x = ((~x0 | ~x1) & (~x0 | ~x2) & (~x0 | ~x3) & (~x0 | ~x4) & (~x0 | ~x5) & (~x0 | ~x6) & (~x0 | ~x7) & (~x0 | ~x8) & (~x0 | ~x9)
                    & (~x1 | ~x2) & (~x1 | ~x3) & (~x1 | ~x4) & (~x1 | ~x5) & (~x1 | ~x6) & (~x1 | ~x7) & (~x1 | ~x8) & (~x1 | ~x9)
                    & (~x2 | ~x3) & (~x2 | ~x4) & (~x2 | ~x5) & (~x2 | ~x6) & (~x2 | ~x7) & (~x2 | ~x8) & (~x2 | ~x9)
                    & (~x3 | ~x4) & (~x3 | ~x5) & (~x3 | ~x6) & (~x3 | ~x7) & (~x3 | ~x8) & (~x3 | ~x9)
                    & (~x4 | ~x5) & (~x4 | ~x6) & (~x4 | ~x7) & (~x4 | ~x8) & (~x4 | ~x9)
                    & (~x5 | ~x6) & (~x5 | ~x7) & (~x5 | ~x8) & (~x5 | ~x9)
                    & (~x6 | ~x7) & (~x6 | ~x8) & (~x6 | ~x9)
                    & (~x7 | ~x8) & (~x7 | ~x9)
                    & (~x8 | ~x9)
                    & (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9))

    constraints_y = ((~y0 | ~y1) & (~y0 | ~y2) & (~y0 | ~y3) & (~y0 | ~y4) & (~y0 | ~y5) & (~y0 | ~y6) & (~y0 | ~y7) & (~y0 | ~y8) & (~y0 | ~y9)
                    & (~y1 | ~y2) & (~y1 | ~y3) & (~y1 | ~y4) & (~y1 | ~y5) & (~y1 | ~y6) & (~y1 | ~y7) & (~y1 | ~y8) & (~y1 | ~y9)
                    & (~y2 | ~y3) & (~y2 | ~y4) & (~y2 | ~y5) & (~y2 | ~y6) & (~y2 | ~y7) & (~y2 | ~y8) & (~y2 | ~y9)
                    & (~y3 | ~y4) & (~y3 | ~y5) & (~y3 | ~y6) & (~y3 | ~y7) & (~y3 | ~y8) & (~y3 | ~y9)
                    & (~y4 | ~y5) & (~y4 | ~y6) & (~y4 | ~y7) & (~y4 | ~y8) & (~y4 | ~y9)
                    & (~y5 | ~y6) & (~y5 | ~y7) & (~y5 | ~y8) & (~y5 | ~y9)
                    & (~y6 | ~y7) & (~y6 | ~y8) & (~y6 | ~y9)
                    & (~y7 | ~y8) & (~y7 | ~y9)
                    & (~y8 | ~y9)
                    & (y0 | y1 | y2 | y3 | y4 | y5 | y6 | y7 | y8 | y9))
    
    sum0 = sum0 & constraints_x & constraints_y
    sum1 = sum1 & constraints_x & constraints_y
    sum2 = sum2 & constraints_x & constraints_y
    sum3 = sum3 & constraints_x & constraints_y
    sum4 = sum4 & constraints_x & constraints_y
    sum5 = sum5 & constraints_x & constraints_y
    sum6 = sum6 & constraints_x & constraints_y
    sum7 = sum7 & constraints_x & constraints_y
    sum8 = sum8 & constraints_x & constraints_y
    sum9 = sum9 & constraints_x & constraints_y
    sum10 = sum10 & constraints_x & constraints_y
    sum11 = sum11 & constraints_x & constraints_y
    sum12 = sum12 & constraints_x & constraints_y
    sum13 = sum13 & constraints_x & constraints_y
    sum14 = sum14 & constraints_x & constraints_y
    sum15 = sum15 & constraints_x & constraints_y
    sum16 = sum16 & constraints_x & constraints_y
    sum17 = sum17 & constraints_x & constraints_y
    sum18 = sum18 & constraints_x & constraints_y

    sum0.ref()
    sum1.ref()
    sum2.ref()
    sum3.ref()
    sum4.ref()
    sum5.ref()
    sum6.ref()
    sum7.ref()
    sum8.ref()
    sum9.ref()
    sum10.ref()
    sum11.ref()
    sum12.ref()
    sum13.ref()
    sum14.ref()
    sum15.ref()
    sum16.ref()
    sum17.ref()
    sum18.ref()

    manager.minimize()

    return {
        0: sum0,
        1: sum1,
        2: sum2,
        3: sum3,
        4: sum4,
        5: sum5,
        6: sum6,
        7: sum7,
        8: sum8,
        9: sum9,
        10: sum10,
        11: sum11,
        12: sum12,
        13: sum13,
        14: sum14,
        15: sum15,
        16: sum16,
        17: sum17,
        18: sum18,
    }
    # fmt: on
