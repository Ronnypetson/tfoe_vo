evo_rpe kitti $1 $2 -va --save_results $3.rot.rpe.zip --align --pose_relation angle_deg
evo_ape kitti $1 $2 -va --save_results $3.rot.ape.zip --align --pose_relation angle_deg

evo_rpe kitti $1 $2 -va --save_results $3.tr.rpe.zip --align --pose_relation trans_part
evo_ape kitti $1 $2 -va --save_results $3.tr.ape.zip --align --pose_relation trans_part
