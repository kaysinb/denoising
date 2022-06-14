def multiplicate(A):
    if len(A) < 2:
        print("the task is not defined when len(A) < 2")
        return None

    product = 1
    zero_position = None
    for i, element in enumerate(A):
        if element != 0:
            product *= element
        else:
            if zero_position is None:
                zero_position = i
            else:
                return [0]*len(A)

    if zero_position is not None:
        out = [0]*len(A)
        out[zero_position] = product
        return out
    return [product // element for element in A]


if __name__ == '__main__':
    # test empty list
    assert (multiplicate([]) is None)
    print(' --- "empty list" test is passed')

    # test list with len = 1
    assert (multiplicate([1]) is None)
    print(' --- "list with len = 1" test is passed')

    # test sample list
    assert (multiplicate([1, 2, 3, 4]) == [24, 12, 8, 6])
    print(' --- "sample list" test is passed')

    # test list with zero
    assert (multiplicate([0, 1, 2]) == [2, 0, 0])
    print(' --- "list with zero" test is passed')

    # test list with several zeros
    assert (multiplicate([0, 1, 0, 2]) == [0, 0, 0, 0])
    print(' --- "list with several zeros" test is passed')

    # test list with negative elements
    assert (multiplicate([-1, 2, -3, 4]) == [-24, 12, -8, 6])
    print(' --- "list with negative elements" test is passed')
