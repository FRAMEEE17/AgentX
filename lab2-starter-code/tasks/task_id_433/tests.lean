<<<<<<< HEAD
#guard isGreater (6) (#[1, 2, 3, 4, 5]) = (true)
#guard isGreater (3) (#[1, 2, 3, 4, 5]) = (false)
#guard isGreater (1) (#[]) = (true)
#guard isGreater (5) (#[5, 5, 5]) = (false)
#guard isGreater (-1) (#[-10, -5, -3]) = (true)
#guard isGreater (-3) (#[-1, -2, -3]) = (false)
#guard isGreater (0) (#[0, -1, -2]) = (false)
=======
#guard isGreater (6) (#[1, 2, 3, 4, 5]) = (true)
#guard isGreater (3) (#[1, 2, 3, 4, 5]) = (false)
#guard isGreater (1) (#[]) = (true)
#guard isGreater (5) (#[5, 5, 5]) = (false)
#guard isGreater (-1) (#[-10, -5, -3]) = (true)
#guard isGreater (-3) (#[-1, -2, -3]) = (false)
#guard isGreater (0) (#[0, -1, -2]) = (false)
>>>>>>> 1e9a9961e8fdb46ae9c2557929ff8e564c9c54ed
#guard isGreater (10) (#[1, 2, 9, 3]) = (true)