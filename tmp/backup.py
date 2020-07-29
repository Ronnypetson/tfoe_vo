# id0 = j - 1
# id1 = j
# id2 = j + 1
# print(id0, id1)
# print(ge[j] - c_[:2, -1])
# print(ge[i] - c_[:2, -1])
# sc = np.linalg.norm(ge[j] - c_[:2, -1])\
#     / np.linalg.norm(ge[i] - c_[:2, -1])
# print(sc)
# input()

# ptid00 = np.random.randint(low=0, high=x[(id0, id1)].shape[1] - 1)
##ptid01 = np.random.randint(low=0, high=x[(id0, id1)].shape[1] - 1)
# ptid01 = (ptid00 + 83) % x[(id0, id1)].shape[1]

# if ptid00 == ptid01:
#    ptid01 = ptid00 + 1

# ptid10 = np.argmin([np.linalg.norm(x_[(id0, id1)][:, ptid00].T - p01)
#                    for p01 in x[(id1, id1 + 1)].T])
# ptid11 = np.argmin([np.linalg.norm(x_[(id0, id1)][:, ptid01].T - p11)
#                    for p11 in x[(id1, id1 + 1)].T])

# X00 = triangulate(x[(id0, id1)][:, [ptid00]], x_[(id0, id1)][:, [ptid00]],
#                  kp._T0[id0], kp.camera_matrix, ge[id0])
# X01 = triangulate(x[(id0, id1)][:, [ptid01]], x_[(id0, id1)][:, [ptid01]],
#                  kp._T0[id0], kp.camera_matrix, ge[id0])

# X10 = triangulate(x[(id1, id1 + 1)][:, [ptid10]], x_[(id1, id1 + 1)][:, [ptid10]],
#                  kp._T0[id1], kp.camera_matrix, ge[id1])
# X11 = triangulate(x[(id1, id1 + 1)][:, [ptid11]], x_[(id1, id1 + 1)][:, [ptid11]],
#                  kp._T0[id1], kp.camera_matrix, ge[id1])

# rs = np.linalg.norm(X00[-1] - X01[-1])\
#     / (np.linalg.norm(X10[-1] - X11[-1]) + 1e-10)

# X01 = triangulate(x[(id0, id1)], x_[(id0, id1)],
#                  kp._T0[id0], kp.camera_matrix, ge[id0])
# X12 = triangulate(x[(id1, id2)], x_[(id1, id2)],
#                  kp._T0[id1], kp.camera_matrix, ge[id1])
# rs = np.mean(np.linalg.norm(X12[-1], axis=0))\
#     / np.mean(np.linalg.norm(X01[-1], axis=0))
# print(rs)
# input()
