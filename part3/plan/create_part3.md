# create_part3

## 任务要求

根据这个文件/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/Project_3_Video_Object_Removal___Inpainting.pdf，完成part 3。

我们计划是用ROSE、DiffuEraser和SAM 3来完成part 3，来解决镜像、倒影问题和影子没有消去的问题。

用的数据路径是这个/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/data/Wild_Video和这个/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/data/DAVIS。DAVIS数据集，先用bmx-trees和tennis，看看效果。

工作区域主要是这个路径/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3，但也可能会修改其他文件夹下的文件。

任务更详细的要求，请参考/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/plan/part3_codex_execution_plan.md。

## 注意事项

1. 我是在slurm集群，所以用GPU要提交作业到计算节点。先用debug分区跑通smoke test进行测试。如果正式作业跑的时间要超过30分钟。我会让我同学去他那个计算资源多的HPC上把完整的作业跑完。他们那个HPC就可以不用sbatch提交作业，直接python就可以用GPU运行程序。
2.  part 1和part 2分别在这些路径下/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part1和/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part2。
3. 更多注意事项，可以参考这个文件/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/plan/Notes_for_Attention.md。
4. 用于在slurm集群提交作业的.sh脚本，存放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts。

