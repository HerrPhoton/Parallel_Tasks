#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

// Преобразование индекса двумерного массива в соответсвующий индекс
// одномерного массива
#define ID(side, row, col) ((((side) + 1) * (row)) + (col))

////////////////////////////////////////////////////////////////////////////////
// Создание динамического массива заданной длины.
// Размер size соответствует общему количеству элементов в сетке.
////////////////////////////////////////////////////////////////////////////////
double *make_grid(size_t size)
{
	double *grid = new double[size];

	return grid;
}

////////////////////////////////////////////////////////////////////////////////
// Вывод динамического массива в виде двумерной сетки.
// Размер side - количество элементов в одной стороне сетки минус 1
////////////////////////////////////////////////////////////////////////////////
void print_grid(double *grid, size_t side)
{
	for (size_t i = 0; i <= side; i++)
	{
		for (size_t j = 0; j <= side; j++)
			std::cout << grid[ID(side, i, j)] << " ";

		std::cout << std::endl;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Заполнение углов сетки заранее заданным значением
// Размер side - количество элементов в одной стороне сетки минус 1
////////////////////////////////////////////////////////////////////////////////
void set_corners(double *grid, size_t side, double* steps)
{
	int upper_left = 10;  // Левый верхний угол
	int upper_right = 20; // Правый верхний угол
	int lower_right = 20; // Правый нижний угол
	int lower_left = 30;  // Левый нижний угол

	grid[ID(side, 0, 0)] = upper_left;
	grid[ID(side, 0, side)] = upper_right;
	grid[ID(side, side, 0)] = lower_right;
	grid[ID(side, side, side)] = lower_left;

	// Расчёт шага для каждой стороны сетки
	steps[0] = (grid[ID(side, 0, side)] - grid[ID(side, 0, 0)]) / side ;
	steps[1] = (grid[ID(side, side, side)] - grid[ID(side, 0, side)]) / side;
	steps[2] = (grid[ID(side, side, side)] - grid[ID(side, side, 0)]) / side;
	steps[3] = (grid[ID(side, side, 0)] - grid[ID(side, 0, 0)]) / side;
}

////////////////////////////////////////////////////////////////////////////////
// Заполнение границ сетки с шагом steps[i]
// Отсчёт ведется с верхней стороны по часовой стрелке
////////////////////////////////////////////////////////////////////////////////
void fill_borders(double *grid, size_t side, double* steps)
{
	for (int i = 0; i < side; i++)
	{
		grid[ID(side, 0, i)] = grid[ID(side, 0, 0)] + steps[0] * i;
		grid[ID(side, i, side)] = grid[ID(side, 0, side)] + steps[1] * i;
		grid[ID(side, side, i)] = grid[ID(side, side, 0)] + steps[2] * i;
		grid[ID(side, i, 0)] = grid[ID(side, 0, 0)] + steps[3] * i;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Нахождение разницы между двумя сетками по модулю
////////////////////////////////////////////////////////////////////////////////
__global__ void find_AbsDiff(double *grid, double *grid_new, double *tmp, size_t rows, size_t cols)
{
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < rows && j > 0 && j < cols)
		tmp[ID(cols, i, j)] =  std::abs(grid_new[ID(cols, i, j)] - grid[ID(cols, i, j)]);
}

////////////////////////////////////////////////////////////////////////////////
// Вычисление нового шага границ сетки через пятиточечный метод расчета
////////////////////////////////////////////////////////////////////////////////
__global__ void calculate_bounds(double *grid, double *grid_new, size_t rows, size_t cols)
{
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i == 0 || i == rows) && j >= 0 && j <= cols)
	{
		grid_new[ID(cols, i, j)] = 0.25 * (grid[ID(cols, i - 1, j)] + grid[ID(cols, i + 1, j)] 
			+ grid[ID(cols, i, j - 1)] + grid[ID(cols, i, j + 1)]);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Вычисление нового шага сетки через пятиточечный метод расчета
////////////////////////////////////////////////////////////////////////////////
__global__ void calculate_grid(double *grid, double *grid_new, size_t rows, size_t cols)
{
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < rows && j > 0 && j < cols)
	{
		grid_new[ID(cols, i, j)] = 0.25 * (grid[ID(cols, i - 1, j)] + grid[ID(cols, i + 1, j)] 
			+ grid[ID(cols, i, j - 1)] + grid[ID(cols, i, j + 1)]);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Основная точка входа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	auto begin = std::chrono::high_resolution_clock::now(); // Фиксация начального времени

	int side      = 0; 	   // Размер стороны целой сетки
	int iters_cnt = 0; 	   // Максимальное заданное количество итераций
	double error  = 0;     // Максимальное заданное значение ошибки
	bool is_show  = false; // Нужно ли выводить сетку в конце работы

	int size        = 0;  // Общее количество элементов сетки
	int passed_iter = 0;  // Количество пройденных итераций

	//Парсинг аргументов
	for (int i = 1; i < argc; i += 2) 
	{
		std::string str = argv[i];

		if (str == "-side") // Количество элементов в одной стороне целой сетки 
		{
			side = std::stoi(argv[i + 1]);	
			size = side * side;

			side--; //Сдвиг для нумерации элементов с нуля
		}

		else if (str == "-error") // Максимальная ошибка между двумя шагами
			error = std::stod(argv[i + 1]);

		else if (str == "-iters") // Максимальное количество между итерациями
			iters_cnt = std::stoi(argv[i + 1]);

		else if (str == "-show") // Требование вывести сетку на экран
			is_show = true;
	}
    
    int rank      = 0;    // Текущий номер процесса
	int num_ranks = 0; 	  // Общее количество процессов

    MPI_Init(&argc, &argv);					   // Инициализация MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Запись номера текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); // Запись общего числа процессов

	cudaSetDevice(rank); // Назначить каждому процессу отдельную GPU

	int side_rank = (side + 1) / num_ranks - 1; // Количество строк в разделенной сетке 
	int start_id  = side_rank * rank;  		    // Индекс, с которого начнет обработку текущий процесс
	
	// Переменные в памяти CPU
	double* h_grid; 						// Часть сетки на текущем шаге
	double* h_grid_new;  					// Часть сетки на новом шаге
	double  h_steps[4];       			  	// Шаг для каждой стороны сетки
	double  h_max_error = error + 1; 	  	// Максимальная достигнутая ошибка

	// Переменные в памяти GPU
	double* d_grid;		 // Часть сетки на текущем шаге
	double* d_grid_new;  // Часть сетки на новом шаге
	double* d_tmp;		 // Разница между сетками на текущем и новом шагах
	double* d_max_error; // Максимальная достигнутая ошибка

	size_t bytes 	  = size * sizeof(double);      			       // Количество байт во всей сетке
	size_t bytes_rank = (side + 1) * (side_rank + 1) * sizeof(double); // Количесво байт сетки для обработки одного процесса

	// Добавить граничные строки для каждого процесса, если они требуются
	if (num_ranks != 1)
		side_rank += (rank > 0 && rank < num_ranks) ? 2 : 1;

	// Инициализация полных сеток в нулевом процессе 
	if (!rank)
	{
		h_grid     = make_grid(size);
		h_grid_new = make_grid(size);

		// Установка значений в углах сетки
		set_corners(h_grid, side, h_steps);
		set_corners(h_grid_new, side, h_steps);

		// Заполнение сторон всех сеток с помощью линейной интерполяции между углами области
		fill_borders(h_grid, side, h_steps);
		fill_borders(h_grid_new, side, h_steps);
	}

	// Выделение памяти под разделенные сетки на GPU
	cudaMalloc(&d_grid, bytes_rank);
	cudaMalloc(&d_grid_new, bytes_rank);
	cudaMalloc(&d_tmp, bytes_rank);
	cudaMalloc(&d_max_error, sizeof(double));

	// Копирование разделенных сеток с CPU на GPU
	cudaMemcpy(d_grid, &h_grid[start_id], bytes_rank, cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_new, &h_grid_new[start_id], bytes_rank, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_max_error, &h_max_error, sizeof(double), cudaMemcpyHostToDevice);

	// Подготовка переменных для вычисления редукции по максимуму
	void *d_tmp_storage = NULL;   // Массив для хранения промежуточных вычислений
	size_t tmp_storage_bytes = 0; // Количество байт для хранения временных данных

	// Найти необходимое количество байт для хранения временных данных
	cub::DeviceReduce::Max(d_tmp_storage, tmp_storage_bytes, d_grid_new, d_max_error, side * side_rank);
	cudaMalloc(&d_tmp_storage, tmp_storage_bytes); // Выделить память под временный массив

	// Инициализация потоков для границ и для внутренних значений
	int leastPriority    = 0;
    int greatestPriority = 0;

	cudaStream_t s_bounds;
	cudaStream_t s_inner;

	cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

	cudaStreamCreateWithPriority(&s_bounds, cudaStreamDefault, greatestPriority);
	cudaStreamCreateWithPriority(&s_inner, cudaStreamDefault, leastPriority);

	// Количество потоков в блоке
	dim3 block_size(8, 8); 

	// Количество блоков в сетке
	dim3 grid_size((side + block_size.x - 1) / block_size.x, (side + block_size.y - 1) / block_size.y);

	int update_step = ((side + 1) / 2); // Количество итераций между обновлениями максимальной ошибки

	while (h_max_error > error && passed_iter < iters_cnt)
	{	
		passed_iter++;

		if (num_ranks > 1)
		{
			calculate_bounds<<<side, 1, 0, s_bounds>>>(d_grid, d_grid_new, side_rank, side);
			cudaStreamSynchronize(s_bounds);
		}

		calculate_grid<<<grid_size, block_size, 0, s_inner>>>(d_grid, d_grid_new, side_rank, side);

		if (passed_iter % update_step == 0)
		{
			find_AbsDiff<<<grid_size, block_size, 0, s_inner>>>(d_grid, d_grid_new, d_tmp, side_rank, side);
			cub::DeviceReduce::Max(d_tmp_storage, tmp_storage_bytes, d_tmp, d_max_error, side * side_rank, s_inner);
			cudaStreamSynchronize(s_inner);
			MPI_Allreduce((void*)d_max_error, (void*)d_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			cudaMemcpyAsync(&h_max_error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost, stream_inner);
		}

		if (num_ranks > 1)
		{
			// Принимаем строки от нижних процессов, если они имеются
			if (rank > 0)
			{
				MPI_Sendrecv(
					&d_grid_new[ID(side, 1, 1)], side - 1, MPI_DOUBLE, rank - 1, 0,
					&d_grid_new[ID(side, 0, 1)], side - 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}

			// Принимаем строки от верхних процессов, если они имеются
			if (rank < num_ranks)
			{
				MPI_Sendrecv(
					&d_grid_new[ID(side, side_rank - 1, 1)], side - 1, MPI_DOUBLE, rank + 1, 0,
					&d_grid_new[ID(side, side_rank, 1)], side - 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}

		}

		std::swap(d_grid, d_grid_new);
	}

	// Копирование данных массивов с GPU на CPU
	cudaMemcpy(h_grid, d_grid, bytes_rank, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	std::cout << "Passed iterations: " << passed_iter << "/" << iters_cnt << std::endl;
	std::cout << "Maximum error: "     << h_max_error << "/" << error     << std::endl;

	auto end = std::chrono::high_resolution_clock::now();									  // Фиксация конечного времени
	double time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(); // Общее время выполнения в микросекундах

	std::string units[] = {"us", "ms", "s"}; // Массив доступных единиц измерения времени
	std::string unit = units[0];            

	// Подбор удобной единицы измерения времени
	for (auto u : units) 
	{
		unit = u;

		// Переход к следующей единице измерения, если при текущей единице
		// значение больше 1000
		if (time > 1000)
			time /= 1000;

		else
			break;
	}

	std::cout << "Total execution time: " << time << " " << unit << std::endl;

	// Вывод сетки на экран
	if (is_show)
		print_grid(h_grid_new, side);

	// Освобождение выделенной памяти
	MPI_Finalize();
	cudaFree(d_grid);
	cudaFree(d_grid_new);
	cudaFree(d_tmp);
	cudaFree(d_max_error);
	delete h_grid;
	delete h_grid_new;

	return 0;
}