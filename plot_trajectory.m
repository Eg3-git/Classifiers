tasks = {'abc', 'cir', 'star', 'www', 'xyz'};
    base_dir = 'Data Collection/u1/u1';
    dir_end = '_ur3e_end_effectors_pose.csv';

    for i = 1:numel(tasks)
        task = tasks{i};
        f_dir = sprintf('%s%s1%s', base_dir, task, dir_end);

        fig = figure;
        ax = axes('parent', fig);

        data = csvread(f_dir, 1, 0);
        np_data = data(:, 2:end);

        plot3(np_data(:, 1), np_data(:, 2), np_data(:, 3));
        title(task);
    end