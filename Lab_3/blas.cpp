#include <iostream>
#include <cstdlib>
#include <cstring>
#include <openacc.h>
#include <cublas_v2.h>
#include <chrono>

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
void set_corners(double *grid, size_t side)
{
	int upper_left = 10;  // Левый верхний угол
	int upper_right = 20; // Правый верхний угол
	int lower_right = 20; // Правый нижний угол
	int lower_left = 30;  // Левый нижний угол

	grid[ID(side, 0, 0)] = upper_left;
	grid[ID(side, 0, side)] = upper_right;
	grid[ID(side, side, 0)] = lower_right;
	grid[ID(side, side, side)] = lower_left;
}

////////////////////////////////////////////////////////////////////////////////
// Основная точка входа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	auto begin = std::chrono::high_resolution_clock::now(); // Фиксация начального времени

	int side = 0;	   	  // Размер стороны сетки
	int iters_cnt = 0; 	  // Максимальное заданное количество итераций
	double error = 0;  	  // Максимальное заданное значение ошибки

	int size = 0; 		  // Общее количество элементов сетки
	bool is_show = false; // Нужно ли выводить сетку в конце работы

	// Парсинг аргументов
	for (int i = 1; i < argc; i += 2) 
	{
		std::string str = argv[i];

		if (str == "-side") // Количество элементов в одной стороне сетки 
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

	double *grid = make_grid(size);     // Сетка на текущем шаге
	double *grid_new = make_grid(size); // Сетка на новом шаге
    double *tmp = make_grid(size);      // Массив, хранящий копию сетки на текущем шаге

	int passed_iter = 0;		  	    // Количество пройденных итераций
	double max_error = error + 1; 		// Максимальная достигнутая ошибка

    cublasHandle_t handle; 									  // Контекст cuBLAS
	cublasCreate(&handle); 									  // Инициализация контекста cuBLAS
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE); // Использование указателей на GPU для функций cuBLAS

	const double alpha = -1; // Скаляр для функций cuBLAS
    int id = 0; 			 // Индекс максимальной ошибки

	double steps[4]; 		 // Шаг для каждой стороны сетки

	#pragma acc data create(tmp[:size], steps[:4]) copyout(grid[:size], grid_new[:size]) copyin(side, size, max_error, alpha, id)
	{
		#pragma acc kernels
		{
			// Установка значений в углах сетки
			set_corners(grid, side);
			set_corners(grid_new, side);
			set_corners(tmp, side);

			// Расчёт шага для каждой стороны сетки
			steps[0] = (grid[ID(side, 0, side)] - grid[ID(side, 0, 0)]) / side ;
			steps[1] = (grid[ID(side, side, side)] - grid[ID(side, 0, side)]) / side;
			steps[2] = (grid[ID(side, side, side)] - grid[ID(side, side, 0)]) / side;
			steps[3] = (grid[ID(side, side, 0)] - grid[ID(side, 0, 0)]) / side;	
		}

		// Заполнение сторон всех сеток с помощью линейной интерполяции между углами области
		#pragma acc parallel loop independent async(1)
			for (size_t i = 1; i < side; i++)
			{ 
				grid[ID(side, 0, i)] = grid[ID(side, 0, 0)] + steps[0] * i;
				grid[ID(side, i, side)] = grid[ID(side, 0, side)] + steps[1] * i;
				grid[ID(side, side, i)] = grid[ID(side, side, 0)] + steps[2] * i;
				grid[ID(side, i, 0)] = grid[ID(side, 0, 0)] + steps[3] * i;
			}

		#pragma acc parallel loop independent async(2)
			for (size_t i = 1; i < side; i++)
			{ 
				grid_new[ID(side, 0, i)] = grid_new[ID(side, 0, 0)] + steps[0] * i;
				grid_new[ID(side, i, side)] = grid_new[ID(side, 0, side)] + steps[1] * i;
				grid_new[ID(side, side, i)] = grid_new[ID(side, side, 0)] + steps[2] * i;
				grid_new[ID(side, i, 0)] = grid_new[ID(side, 0, 0)] + steps[3] * i;
			}
			
		#pragma acc parallel loop independent async(3)
			for (size_t i = 1; i < side; i++)
			{
                tmp[ID(side, 0, i)] = grid_new[ID(side, 0, 0)] + steps[0] * i;
				tmp[ID(side, i, side)] = grid_new[ID(side, 0, side)] + steps[1] * i;
				tmp[ID(side, side, i)] = grid_new[ID(side, side, 0)] + steps[2] * i;
				tmp[ID(side, i, 0)] = grid_new[ID(side, 0, 0)] + steps[3] * i;
			}

		#pragma acc wait

		while (max_error > error && passed_iter < iters_cnt)
		{
			#pragma acc parallel loop collapse(2) independent present(grid, grid_new)
				for (size_t i = 1; i < side; i++)
					for (size_t j = 1; j < side; j++)
                    {
                        // Пятиточечный метод расчета сетки
						grid_new[ID(side, i, j)] = 0.25 * (grid[ID(side, i - 1, j)] + grid[ID(side, i + 1, j)] 
							+ grid[ID(side, i, j - 1)] + grid[ID(side, i, j + 1)]);
                    }


			passed_iter++;
			
			// Вычисление ошибки, когда количество итераций кратно половине
			// размера стороны сетки или при прохождении заданного 
			// количества итераций	  
            if (passed_iter % ((side + 1) / 2) == 0 || passed_iter == iters_cnt)
			{
				//Вычисление редукции с помощью cuBLAS
				#pragma acc data present(grid, grid_new, tmp, alpha, id)
				#pragma acc host_data use_device(grid, grid_new, tmp, alpha, id)
				{
					cublasDcopy(handle, size, grid, 1, tmp, 1);			    // Сохранить текущее состояние сетки
					cublasDaxpy(handle, size, &alpha, grid_new, 1, tmp, 1); // Вычисление ошибки между текущей и прошлой итерациями	
					cublasIdamax(handle, size, tmp, 1, &id);                // Вернуть индекс максимального по модулю значения

					#pragma acc kernels
						max_error = std::abs(tmp[id - 1]); // Сохранить значение максимальной ошибки	
				}

				#pragma acc update host(max_error) //Обновление значения max_error в памяти CPU
			}		
				
			// Копирование массива путем смены указателей на CPU
			// и дальнейшим обновлением адресов на GPU
			std::swap(grid, grid_new);

			acc_attach((void**)&grid);
			acc_attach((void**)&grid_new);
		}
	}

	std::cout << "Passed iterations: " << passed_iter << "/" << iters_cnt << std::endl;
	std::cout << "Maximum error: " << max_error << "/" << error << std::endl;

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
		print_grid(grid, side);

	// Освобождение выделенной памяти
    cublasDestroy(handle);
	delete grid;
	delete grid_new;

	return 0;
}