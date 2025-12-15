/*
 * ============================================================================
 * PROBLEMA DE PROGRAMAÇÃO DE HORÁRIOS UNIVERSITÁRIOS (TIMETABLING)
 * Solução usando SIMULATED ANNEALING
 * ============================================================================
 * 
 * Este programa resolve o University Course Timetabling Problem (UCTP)
 * do ITC 2007 usando a metaheurística Simulated Annealing.
 * 
 * OBJETIVO: Alocar disciplinas a períodos e salas respeitando restrições
 * REPRESENTAÇÃO: Matriz [períodos x salas] onde cada célula contém uma disciplina
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// ============================================================================
// CONSTANTES GLOBAIS
// ============================================================================

#define SIZE 100          // Tamanho máximo para strings (nomes, etc)
#define SAIDA 1024        // Tamanho do buffer de saída
#define HISTORICO 10      // Quantidade de soluções a guardar no histórico

// ============================================================================
// RESTRIÇÕES DO PROBLEMA
// ============================================================================

/*
 * RESTRIÇÕES GRAVES (Hard Constraints) - Penalização: 1.000.000 pontos
 * 
 * R1 - AULAS: Todas as aulas devem ser agendadas no número correto
 *      Violação: aula não agendada ou agendada em excesso
 * 
 * R2 - CONFLITOS: Aulas do mesmo curso ou professor não podem coincidir
 *      Violação: duas ou mais aulas conflitantes no mesmo período
 * 
 * R3 - OCUPAÇÃO: Duas aulas não podem usar a mesma sala simultaneamente
 *      Violação: múltiplas aulas na mesma sala/período (garantida pela estrutura)
 * 
 * R4 - DISPONIBILIDADE: Respeitar indisponibilidade dos professores
 *      Violação: aula agendada quando professor não está disponível
 * 
 * R10 - TIPO DE SALA: Respeitar os tipo de de sala para cada disciplina
 *      Violação: aula agendada em uma sala com tipo errado
 * 
 * R11 - DISTRIBUIÇÃO: Aulas da mesma disciplinas devem ser em dias distintos
 *      Violação: aulas da mesma disciplina no mesmo dia
 */

/*
 * RESTRIÇÕES LEVES (Soft Constraints) - Penalizações variadas
 * 
 * R5 - DIAS MÍNIMOS: Distribuir aulas ao longo dos dias (5 pontos/dia faltante)
 *      Violação: disciplina não atinge o número mínimo de dias
 * 
 * R6 - COMPACIDADE: Aulas do mesmo curso devem ser adjacentes (2 pontos/aula isolada)
 *      Violação: aula isolada sem outra do mesmo curso antes ou depois
 * 
 * R7 - CAPACIDADE: Sala deve comportar todos os alunos (1 ponto/aluno extra)
 *      Violação: número de alunos excede capacidade da sala
 * 
 * R8 - ESTABILIDADE: Todas as aulas da disciplina na mesma sala (1 ponto/sala extra)
 *      Violação: disciplina usa mais de uma sala diferente
 * 
 * R9 - Professores não podem dar aula em mais de 2 dias da semana (5 pontos/dia extra)
 *      Violação: Professor que dá aula em mais de 2 dias na semana (3+ dias)
 */

// ============================================================================
// ESTRUTURAS DE DADOS
// ============================================================================

/*
 * MATRIZ: Representa uma solução completa do problema
 * - n: matriz [período][sala] = id_disciplina (-1 se vazio)
 * - fo: valor da função objetivo (soma de todas as penalidades)
 */
typedef struct matriz{
	int fo;              // Função Objetivo (fitness da solução)
	int** n;             // Matriz de alocação [total_periodos][salas]
}Matriz;

/*
 * PROFESSORES: Armazena informações dos professores
 */
typedef struct professores{
	char nome[SIZE];     // Nome do professor
}Professores;

/*
 * DISCIPLINA: Representa uma disciplina a ser agendada
 */
typedef struct disciplina{
	char nome[SIZE];     // Nome da disciplina
	int* cursos;         // Vetor indicando a quais cursos pertence [cursos]
	int  prof;           // ID do professor que ministra
	char profe[SIZE];    // Nome do professor (redundante para impressão)
	int  aulas;          // Número de aulas que devem ser agendadas
	int  minDias;        // Número mínimo de dias que a disciplina deve aparecer (R5)
	int  alunos;         // Número de alunos matriculados (para R7)
	int  tipo_sala;      // Tipo de sala
}Disciplina;

/*
 * SALA: Representa uma sala de aula
 */
typedef struct sala{
	char nome[SIZE];     // Nome/identificador da sala
	int capacidade;      // Capacidade máxima de alunos
	int  tipo_sala;      // Tipo de sala
}Sala;

/*
 * CURSO: Representa um curso/currículo
 */
typedef struct curso{
	char nome[SIZE];     // Nome do curso
	int  qtDisc;         // Quantidade de disciplinas no curso
	int* disciplina;     // Vetor com IDs das disciplinas [qtDisc]
}Curso;

/*
 * RESTRICAO: Representa uma restrição de indisponibilidade (R4)
 */
typedef struct restricao{
	int disciplina;      // ID da disciplina restrita
	int dia;             // Dia da semana indisponível
	int per;             // Período do dia indisponível
}Restricao;

// ============================================================================
// VARIÁVEIS GLOBAIS
// ============================================================================

// Controle de execução
int execucao;                              // Número da execução atual
int rotina = 0;                            // Contador de rotinas executadas
int programa;                              // Contador de programas/instâncias
int mat_solucao_tempo[HISTORICO][2];      // Histórico [i][0]=FO, [i][1]=tempo
int aux_mat = 0;                           // Índice circular para histórico
int num_exec = 1;                          // Número total de execuções planejadas

// Dados do problema (lidos do arquivo)
Professores *prof;        // Vetor de professores
Disciplina *disc;         // Vetor de disciplinas
Sala *sala;               // Vetor de salas
Curso *curso;             // Vetor de cursos
Restricao *restricao;     // Vetor de restrições de indisponibilidade

// Variáveis de controle das restrições
int* r1;                  // [disciplinas] - Conta aulas agendadas por disciplina (R1)
int** r21;                // [total_periodos][professores] - Aulas por professor/período (R2)
int** r22;                // [total_periodos][cursos] - Disciplinas por curso/período (R2)
int** r5;                 // [disciplinas][dias] - Marca dias com aula de cada disciplina (R5)
int* r8;                  // [disciplinas] - Primeira sala usada por disciplina (R8)
int** r9;                 // [professores][dias] - Marca dias com aula de cada disciplina (R9)
int** r11;                 // [dias][disciplinas] - Marca quais disciplinas tem no dia (R11)

int restricoes_violadas[12];  // Contador de violações por tipo de restrição
int* posicao_restricao[2];   // [0]=início, [1]=fim das restrições por disciplina
int ** dias_ocupados_integral;     // [professores][dias] - recebido da integral
int  usar_restricao_integral = 0;  // Flag: 1 = usa dias_ocupados, 0 = normal
int num_profs_da_integral = 0;  //Número de professores no integral

// Variáveis auxiliares para movimentos direcionados
int* aux_mov_r21;         // Guarda posição de conflito de professor
int* aux_mov_r22;         // Guarda posição de conflito de curso
int* aux_mov_r4;          // Guarda posição de violação de disponibilidade
int* aux_mov_r5;          // Guarda número de dias com aula (para R5)
int** aux_mov_r6;         // [total_periodos][salas] - Disciplinas isoladas (R6)
int** aux_mov_r7;         // [disciplinas][2] - [0]=excesso, [1]=posição (R7)
int* aux_mov_r8;          // Posições de violação de estabilidade (R8)
int* aux_mov_r9;          // Guarda professores com aula em mais de três dias (R9)
int** aux_mov_r10;        // [disciplinas][2] - [0]=tipo correto, [1]=posição (R7)
int** aux_mov_r11;         // [total_periodos][salas] - Disciplinas isoladas (R6)


// Parâmetros da instância
char  nome[SIZE];         // Nome da instância
int  professores;         // Número de professores
int  disciplinas;         // Número de disciplinas
int        salas;         // Número de salas
int         dias;         // Número de dias da semana
int periodos_dia;         // Número de períodos por dia
int total_periodos;       // Total de períodos (dias × periodos_dia)
int       cursos;         // Número de cursos
int   restricoes;         // Número de restrições de indisponibilidade


// Parâmetros do Simulated Annealing
float Tinicial;           // Temperatura inicial
float T;                  // Temperatura atual
float Tfinal;             // Temperatura final (critério de parada)
float alpha;              // Taxa de resfriamento (0 < alpha < 1)
int maxIteracoes;         // Número de iterações por temperatura

// ============================================================================
// FUNÇÕES AUXILIARES BÁSICAS
// ============================================================================

/*
 * MODULO: Calcula o módulo da diferença entre dois números
 * Usado para calcular violações de R1 (diferença entre aulas agendadas e requeridas)
 */
int modulo(int n1, int n2){
	if(n1 > n2) return n1 - n2;  // Se n1 maior, retorna n1 - n2
	else return n2 - n1;          // Caso contrário, retorna n2 - n1
}

/*
 * NUMDISCIPLINA: Retorna o ID de uma disciplina dado seu nome
 * Retorna -1 se não encontrada
 */
int numDisciplina(char *nome){
	int i;
	// Percorre todas as disciplinas buscando o nome
	for(i = 0; i < disciplinas; i++) 
		if(strcmp(nome, disc[i].nome) == 0) 
			return i;  // Retorna o índice se encontrou
	return -1;  // Retorna -1 se não encontrou
}

/*
 * NUMPROF: Retorna o ID de um professor dado seu nome
 * Retorna -1 se não encontrado
 */
int numProf(char *nome){
	int i;
	// Percorre todos os professores buscando o nome
	for(i = 0; i < disciplinas; i++) 
		if(strcmp(nome, prof[i].nome) == 0) 
			return i;  // Retorna o índice se encontrou
	return -1;  // Retorna -1 se não encontrou
}

/*
 * RANDOMDOUBLE: Gera número aleatório double no intervalo [inicio, fim)
 * Usado para o critério de Metropolis no SA
 */
double randomDouble(double inicio, double fim){
	// rand() % 10000 gera número de 0 a 9999
	// Divide por 10000.0 para obter valor entre 0 e 1
	// Multiplica pelo intervalo e soma ao início
	return ((double) (rand() % 10000) / 10000.0) * (fim-inicio) + inicio;
}

/*
 * RANDOMINT: Gera número aleatório inteiro no intervalo [inicio, fim]
 * Usado para gerar movimentos aleatórios
 */
int randomInt(int inicio, int fim){
	// Usa randomDouble e converte para int
	return (int) randomDouble(0, fim - inicio + 1.0) + inicio;
}

// ============================================================================
// LEITURA DO ARQUIVO DE ENTRADA
// ============================================================================

/*
 * LEARQUIVOS: Lê arquivo de entrada no formato ITC 2007
 * 
 * Formato esperado:
 * - Name: <nome>
 * - Courses: <disciplina> <professor> <aulas> <minDias> <alunos>
 * - Rooms: <sala> <capacidade>
 * - Curricula: <curso> <qtDisc> <disc1> <disc2> ...
 * - Unavailability_Constraints: <disciplina> <dia> <período>
 * 
 * Retorna 1 se sucesso, 0 se erro
 */
int leArquivos(char *arquivo){
	strcpy(nome, "");              // Inicializa variável de nome
	FILE *fp;                      // Ponteiro para arquivo
	int i, c, aux;                 // Contadores
	int tipo = 0;                  // Controla tipo de entrada sendo lida
	char x[SIZE * SIZE];           // Buffer de leitura
	char *lc;                      // Ponteiro para tokenização
	char char_aux[SIZE];           // Buffer auxiliar

	strcpy(x, "");                 // Inicializa buffer
    
    // SE É SEGUNDA CONSTRUÇÃO (ou mais): Libera estruturas antigas    
    if(rotina > 0){
        // Libera sub-estruturas de disc
        for(i = 0; i < disciplinas; i++){
            free(disc[i].cursos);
        }
        free(disc);
        
        // Libera sub-estruturas de curso
        for(i = 0; i < cursos; i++){
            free(curso[i].disciplina);
        }
        free(curso);
        
        // Libera outras estruturas
        free(prof);
        free(sala);
        free(restricao);
        free(posicao_restricao[0]);
        free(posicao_restricao[1]);
        
        // Libera matrizes bidimensionais
        for(i = 0; i < total_periodos; i++){
            free(r21[i]);
            free(r22[i]);
            free(aux_mov_r6[i]);
            free(aux_mov_r11[i]);
        }
        free(r21);
        free(r22);
        free(aux_mov_r6);
        free(aux_mov_r11);
        
        for(i = 0; i < disciplinas; i++){
            free(r5[i]);
            free(aux_mov_r7[i]);
            free(aux_mov_r10[i]);
        }
        free(r5);
        free(aux_mov_r7);
        free(aux_mov_r10);
        
        for(i = 0; i < professores; i++){
            free(r9[i]);
        }
        free(r9);
        
        for(i = 0; i < dias; i++){
            free(r11[i]);
        }
        free(r11);
        
        // Libera vetores auxiliares
        free(aux_mov_r21);
        free(aux_mov_r22);
        free(aux_mov_r4);
        free(aux_mov_r5);
        free(aux_mov_r8);
        free(aux_mov_r9);
        free(r1);
        free(r8);
    }

	// Tenta abrir o arquivo
	fp = fopen(arquivo, "r");      // Modo leitura
	if(!fp)	{
		printf("Erro na abertura do arquivo de entrada.\n\n");
		getchar();
		return 0;  // Retorna erro
	}
	else{
		i = fgetc(fp);             // Lê primeiro caractere
		tipo++;                    // Começa no tipo 1 (nome)
		
		// Loop principal de leitura
		while(!feof(fp)){
			// Verifica fim da linha
			if((char)i == '\n'){
				// === TIPO 1: Nome da instância ===
				if(tipo == 1){
					sscanf(x, "%*s %s", nome);  // Ignora "Name:" e pega o nome
					tipo++;
				}
				// === TIPO 2: Quantidade de disciplinas ===
				else if(tipo == 2){
					sscanf(x, "%*s %d", &disciplinas);  // Ignora "Courses:" e pega número
					// Aloca memória para disciplinas e professores
					disc = (Disciplina*) malloc(disciplinas * sizeof(Disciplina));
					prof = (Professores*) malloc(disciplinas * sizeof(Professores));
					tipo++;
				}
				// === TIPO 3: Quantidade de salas ===
				else if(tipo == 3){
					sscanf(x, "%*s %d", &salas);  // Ignora "Rooms:" e pega número
					sala = (Sala*) malloc(salas * sizeof(Sala));  // Aloca memória
					tipo++;
				}
				// === TIPO 4: Quantidade de dias ===
				else if(tipo == 4){
					sscanf(x, "%*s %d", &dias);  // Ignora "Days:" e pega número
					tipo++;
				}
				// === TIPO 5: Períodos por dia ===
				else if(tipo == 5){
					sscanf(x, "%*s %d", &periodos_dia);  // Ignora "Periods_per_day:" e pega número
					tipo++;
				}
				// === TIPO 6: Quantidade de cursos ===
				else if(tipo == 6){
					sscanf(x, "%*s %d", &cursos);  // Ignora "Curricula:" e pega número
					curso = (Curso*) malloc(cursos * sizeof(Curso));  // Aloca memória
					tipo++;
				}
				// === TIPO 7: Quantidade de restrições ===
				else if(tipo == 7){
					sscanf(x, "%*s %d", &restricoes);  // Ignora "Constraints:" e pega número
					restricao = (Restricao*) malloc(restricoes * sizeof(Restricao));  // Aloca memória
					
					// Aloca vetores para marcar início e fim de restrições por disciplina
					posicao_restricao[0] = (int*) malloc(disciplinas * sizeof(int));
					posicao_restricao[1] = (int*) malloc(disciplinas * sizeof(int));
					
					// Inicializa com -1 (sem restrições)
					for(i = 0; i < disciplinas; i++){
						posicao_restricao[0][i] = -1;
						posicao_restricao[1][i] = -1;
					}
					tipo++;
				}
				// === TIPO 10: Lendo disciplinas ===
				else if(tipo == 10){
					if(c < disciplinas){  // Ainda há disciplinas para ler
						// Lê nome do professor
						sscanf(x, "%*s %s", char_aux);
						aux = numProf(char_aux);
						
						// Se professor ainda não foi catalogado
						if(aux == -1){
							aux = professores;
							sscanf(char_aux, "%s", prof[professores].nome);
							professores++;  // Incrementa contador de professores
						}

						// Lê dados da disciplina
						sscanf(x, "%s %*s %d %d %d %d", disc[c].nome, &disc[c].aulas,
						       &disc[c].minDias, &disc[c].alunos, &disc[c].tipo_sala);
						disc[c].prof = aux;  // Associa professor
						strcpy(disc[c].profe, prof[aux].nome);  // Copia nome do professor
						
						// Aloca e inicializa vetor de cursos
						disc[c].cursos = (int*) malloc(cursos * sizeof(int));
						for(i = 0; i < cursos; i++) 
							disc[c].cursos[i] = 0;  // 0 = não pertence ao curso
						
						c++;  // Próxima disciplina
					}
					else{tipo = 0; c = 0;}  // Terminou, reseta tipo
				}
				// === TIPO 20: Lendo salas ===
				else if(tipo == 20){
					if(c < salas){  // Ainda há salas para ler
						sscanf(x, "%s %d %d", sala[c].nome, &sala[c].capacidade, &sala[c].tipo_sala);
						c++;  // Próxima sala
					}
					else{tipo = 0; c = 0;}  // Terminou, reseta tipo
				}
				// === TIPO 30: Lendo cursos ===
				else if(tipo == 30){
					if(c < cursos){  // Ainda há cursos para ler
						// Lê nome e quantidade de disciplinas
						sscanf(x, "%s %d", curso[c].nome, &curso[c].qtDisc);
						curso[c].disciplina = (int*) malloc(curso[c].qtDisc * sizeof(int));
						
						aux = 0;
						// Tokeniza a linha para pegar todas as disciplinas
						lc = strtok(x, " ");
						while(lc != NULL){
							if(aux >= 2){  // Começa a listagem das disciplinas (após nome e qtDisc)
								sscanf(lc, "%s", char_aux);
								curso[c].disciplina[aux - 2] = numDisciplina(char_aux);
								// Marca que a disciplina pertence a este curso
								disc[numDisciplina(char_aux)].cursos[c] = 1;
							}
							lc = strtok(NULL, " ");  // Próximo token
							aux++;
						}
						c++;  // Próximo curso
					}
					else{tipo = 0; c = 0;}  // Terminou, reseta tipo
				}
				// === TIPO 40: Lendo restrições ===
				else if(tipo == 40){
					if(c < restricoes){  // Ainda há restrições para ler
						sscanf(x, "%s %d %d", char_aux, &restricao[c].dia, &restricao[c].per);
						restricao[c].disciplina = numDisciplina(char_aux);
						
						// Marca início da lista de restrições desta disciplina
						if(posicao_restricao[0][numDisciplina(char_aux)] == -1)
							posicao_restricao[0][numDisciplina(char_aux)] = c;
						
						// Atualiza fim da lista de restrições desta disciplina
						posicao_restricao[1][numDisciplina(char_aux)] = c;
						c++;  // Próxima restrição
					}
					else{tipo = 0; c = 0;}  // Terminou, reseta tipo
				}
				// Identifica seções do arquivo
				else if(strcmp(x, "COURSES:") == 0)	{tipo = 10; c = 0;}
				else if(strcmp(x, "ROOMS:") == 0)	{tipo = 20; c = 0;}
				else if(strcmp(x, "CURRICULA:") == 0)	{tipo = 30; c = 0;}
				else if(strcmp(x, "UNAVAILABILITY_CONSTRAINTS:") == 0){	tipo = 40; c = 0;}
				
				strcpy(x, "");  // Limpa buffer para próxima linha
			}
			else sprintf(x, "%s%c", x, i); 	// Acrescenta caractere na string
			
			i = fgetc(fp);  // Lê próximo caractere
		}
	}
	fclose(fp);  // Fecha arquivo

	// Calcula total de períodos
	total_periodos = periodos_dia * dias;

	// Vetores auxiliares para movimentos direcionados
	aux_mov_r21 = (int*) malloc(disciplinas * sizeof(int));
	aux_mov_r22 = (int*) malloc(disciplinas * sizeof(int));
	aux_mov_r4 = (int*) malloc(disciplinas * sizeof(int));
	aux_mov_r5 = (int*) malloc(disciplinas * sizeof(int));
	aux_mov_r6 = (int**) malloc(total_periodos * sizeof(int *));
	aux_mov_r7 = (int**) malloc(disciplinas * sizeof(int *));
	aux_mov_r8 = (int*) malloc(periodos_dia*dias * sizeof(int));
	aux_mov_r9 = (int*) malloc(professores * sizeof(int));
	aux_mov_r10 = (int**) malloc(disciplinas * sizeof(int *));
	aux_mov_r11 = (int**) malloc(total_periodos * sizeof(int*));
	

	// Vetores de controle de restrições
	r1 =  (int*) malloc(disciplinas * sizeof(int));
	r21 = (int**) malloc(total_periodos * sizeof(int *));
	r22 = (int**) malloc(total_periodos * sizeof(int *));
	r5 = (int**) malloc(disciplinas * sizeof(int *));
	r8 = (int*) malloc(disciplinas * sizeof(int));
	r9 = (int**) malloc(professores * sizeof(int*));
	r11 = (int**) malloc(dias * sizeof(int*));
	
	// Aloca segunda dimensão das matrizes
	for(i = 0; i < total_periodos; i++){
		r21[i] = (int*) malloc(professores * sizeof(int));
		r22[i] = (int*) malloc(cursos * sizeof(int));
		aux_mov_r6[i] = (int*) malloc(salas * sizeof(int));
	}
	for(i = 0; i < disciplinas; i++){
		r5[i] = (int*) malloc(dias * sizeof(int));
		aux_mov_r7[i] = (int*) malloc(2 * sizeof(int));
		aux_mov_r10[i] = (int*) malloc(2 * sizeof(int)); 
	}
	for(i = 0; i < professores; i++){
		r9[i] = (int*) malloc(dias * sizeof(int));
	}
	for(i = 0; i < dias; i++){
		r11[i] = (int*) malloc(disciplinas * sizeof(int));
	}
	for(i = 0; i < total_periodos; i++){
		aux_mov_r11[i] = (int*) malloc(salas * sizeof(int));
	}
	
	return 1;  // Sucesso
}

// ============================================================================
// FUNÇÕES DE IMPRESSÃO E SAÍDA
// ============================================================================

/*
 * IMPRIMESOLUCAO: Exibe a solução em formato de grade
 * Mostra período x sala com as disciplinas alocadas
 */
void imprimeSolucao(Matriz matriz){
	int i, j;
	printf("\n");

	// Cabeçalho: nomes das salas
	printf("[Dia/Per");
	for(j = 0; j < salas; j++) 
		printf("|%s\t", sala[j].nome);
	printf("|]\n");
	
	// Linhas: cada período
	for(i = 0; i < total_periodos; i++){
		// Imprime dia e período (i/periodos_dia = dia, i%periodos_dia = período)
		printf("[ %d, %d\t", i/periodos_dia, i%periodos_dia);
		
		// Imprime cada sala
		for(j = 0; j < salas; j++){
			if(matriz.n[i][j] < 0) 
				printf("|-(%d)-\t", matriz.n[i][j]);  // Vazio
			else 
				printf("|%s\t", disc[matriz.n[i][j]].nome);  // Disciplina
		}
		printf("|]\n");
	}

	// Exibe valor da função objetivo
	printf("\n***FO = %d***\n", matriz.fo);
}

// ============================================================================
// FUNÇÕES DE VERIFICAÇÃO DE RESTRIÇÕES
// ============================================================================

/*
 * RESTRICAOR4: Verifica se há restrição de indisponibilidade
 * Retorna 1 se a disciplina não pode ser alocada neste período, 0 caso contrário
 * 
 * Esta função verifica a RESTRIÇÃO R4 (Disponibilidade do professor)
 */
int restricaoR4(int dis, int per){
	int dia = per/periodos_dia;       // Extrai dia do período
	int diaPer = per%periodos_dia;    // Extrai período do dia
	int i;

	// Se a disciplina não possui restrições
	if(posicao_restricao[0][dis] == -1) 
		return 0;
	
	// As restrições estão organizadas por disciplina
	// Busca apenas no intervalo de restrições desta disciplina
	for(i = posicao_restricao[0][dis]; i <= posicao_restricao[1][dis]; i++){
		// Verifica se é a disciplina certa, no dia e período certos
		if((restricao[i].disciplina == dis) && 
		   (restricao[i].dia == dia) && 
		   (restricao[i].per == diaPer))
			return 1;  // Existe restrição - VIOLAÇÃO!
	}
	return 0;  // Não existe restrição - OK
}

/*
 * RESTRICAOR6: Calcula penalidade por falta de compacidade
 * Verifica se a disciplina está isolada (sem adjacentes do mesmo curso)
 * Retorna a penalidade total (2 pontos por curso em que está isolada)
 * 
 * Esta função verifica a RESTRIÇÃO R6 (Compacidade do currículo)
 */
int restricaoR6(Matriz matriz, int dis, int per, int sal){
	int i, j, penalidade, ok;

	penalidade = 0;
	
	// Para cada curso que esta disciplina pertence
	for(i = 0; i < cursos; i++){
		ok = 0;
		
		// Se a disciplina pertence ao curso i
		if(disc[dis].cursos[i] == 1){
			// Busca disciplinas adjacentes do mesmo curso
			for(j = 0; j < salas; j++){
				// Verifica período SEGUINTE (se não for o último período do dia)
				if(((per % periodos_dia) < (periodos_dia - 1)) && (ok == 0)){
					if(matriz.n[per + 1][j] != -1){
						// Se a disciplina seguinte também pertence ao curso
						if(disc[matriz.n[per + 1][j]].cursos[i] == 1){
							ok = 1;        // Encontrou adjacente
							j = salas;     // Encerra busca
						}
					}
				}
				// Verifica período ANTERIOR (se não for o primeiro período do dia)
				if(((per % periodos_dia) > 0) && (ok == 0)){
					if(matriz.n[per - 1][j] != -1){
						// Se a disciplina anterior também pertence ao curso
						if(disc[matriz.n[per - 1][j]].cursos[i] == 1) {
							ok = 1;        // Encontrou adjacente
							j = salas;     // Encerra busca
						}
					}
				}
			}
			// Se não encontrou adjacente para o curso i
			if(ok == 0){
				penalidade += 2;  // Penalidade de 2 pontos
				restricoes_violadas[6]++;
				aux_mov_r6[per][sal] = dis;  // Guarda para possível movimento
			}
		}
	}
	return penalidade;
}

// ============================================================================
// FUNÇÕES AUXILIARES PARA MANIPULAÇÃO DE VETORES
// ============================================================================

/*
 * SETVETOR: Inicializa todos os elementos de um vetor com um valor
 */
void setVetor(int *vetor, int tamanho, int valor){
	int i;
	for(i = 0; i < tamanho; i++) 
		vetor[i] = valor;
}

/*
 * SETMATRIZ: Inicializa todos os elementos de uma matriz com um valor
 */
void setMatriz(int **vetor, int linha, int coluna, int valor){
	int i, j;
	for(i = 0; i < linha; i++){
		for(j = 0; j < coluna; j++)
			vetor[i][j] = valor;
	}
}

/*
 * SOMAVETOR: Soma todos os elementos de um vetor
 */
int somaVetor(int *vetor, int tamanho){
	int soma, i;
	soma = 0;
	for(i = 0; i < tamanho; i++)
		soma += vetor[i];
	return soma;
}

// ============================================================================
// FUNÇÃO OBJETIVO - CÁLCULO DAS PENALIDADES
// ============================================================================

/*
 * CALCULA_FO: Calcula o valor da função objetivo (fitness)
 * 
 * A função objetivo é a SOMA de todas as penalidades:
 * - Restrições graves: 1.000.000 pontos por violação
 * - Restrições leves: 1-5 pontos por violação
 * 
 * OBJETIVO: Minimizar fo (idealmente fo = 0)
 * 
 * Esta é a função mais importante do algoritmo, pois define
 * a qualidade de cada solução.
 */
int calcula_FO(Matriz matriz){
	int fo = 0;  // Inicializa função objetivo
	int i, j, k;
	int r5_aux, sub_r7;

	// ========================================================================
	// INICIALIZAÇÃO: Zera todas as estruturas de controle
	// ========================================================================
	
	setMatriz(r21, total_periodos, professores, 0);   // Zera conflitos de professor
	setMatriz(r22, total_periodos, cursos, 0);        // Zera conflitos de curso
	setMatriz(r5, disciplinas, dias, 0);              // Zera contagem de dias
    setMatriz(r9, professores, dias, 0);
    

    if(usar_restricao_integral && dias_ocupados_integral != NULL){
        // Copia apenas os professores que existem na integral
        for(i = 0; i < professores; i++){
            for(j = 0; j < dias; j++){
                if(i < num_profs_da_integral){
                    r9[i][j] = dias_ocupados_integral[i][j];
                }
            }
        }
    }
	setMatriz(r11, dias, disciplinas, 0);              // Zera contagem de dias com disciplinas (R9)
	setMatriz(aux_mov_r6, total_periodos, salas, -1); // Zera auxiliar R6
	setMatriz(aux_mov_r7, disciplinas, 2, -1);        // Zera auxiliar R7
	setMatriz(aux_mov_r10, disciplinas, 2, -1);        // Zera auxiliar R10
	setMatriz(aux_mov_r11, total_periodos, salas, -1);;// Zera auxiliar R11

	setVetor(aux_mov_r21, disciplinas, -1);  // Zera auxiliar R2 (professor)
	setVetor(aux_mov_r22, disciplinas, -1);  // Zera auxiliar R2 (curso)
	setVetor(aux_mov_r4, disciplinas, -1);   // Zera auxiliar R4
	setVetor(aux_mov_r5, disciplinas, -1);   // Zera auxiliar R5
	setVetor(aux_mov_r8, total_periodos, -1); // Zera auxiliar R8
    setVetor(aux_mov_r9, professores, -1);   // Zera auxiliar R9
	setVetor(r1, disciplinas, 0);            // Zera contagem de aulas (R1)
	setVetor(r8, disciplinas, -1);           // Zera primeira sala (R8)
	setVetor(restricoes_violadas, 12, -1);    // Zera contador de violações
	setMatriz(r11, dias, disciplinas, 0);	  // Zera auxiliar R11

	// ========================================================================
	// LOOP PRINCIPAL: Percorre toda a matriz e conta violações
	// ========================================================================
	
	for(i = 0; i < total_periodos; i++){      // Para cada período
		for(j = 0; j < salas; j++){           // Para cada sala
			if(matriz.n[i][j] != -1){        // Se há aula alocada
				
				// ============================================================
				// R1: Conta quantas vezes cada disciplina foi alocada
				// ============================================================
				r1[matriz.n[i][j]]++;
				
				// ============================================================
				// R2: Verifica conflitos de PROFESSOR
				// ============================================================
				r21[i][disc[matriz.n[i][j]].prof]++;  // Incrementa contador
				// Se professor tem mais de uma aula no mesmo período
				if(r21[i][disc[matriz.n[i][j]].prof] > 1) 
					aux_mov_r21[matriz.n[i][j]] = (j * total_periodos) + i;  // Guarda posição
				
				// ============================================================
				// R2: Verifica conflitos de CURSO
				// ============================================================
				for(k = 0; k < cursos; k++){
					// Se a disciplina pertence ao curso k
					if(disc[matriz.n[i][j]].cursos[k] == 1)	
						r22[i][k]++;  // Incrementa contador
					// Se o curso tem mais de uma disciplina no mesmo período
					if(r22[i][k] > 1) 
						aux_mov_r22[matriz.n[i][j]] = (j * total_periodos) + i;  // Guarda posição
				}
				
				// ============================================================
				// R4: Verifica disponibilidade do professor
				// ============================================================
				if(restricaoR4(matriz.n[i][j], i) == 1){
					aux_mov_r4[matriz.n[i][j]] = (j * total_periodos) + i;  // Guarda posição
					restricoes_violadas[4]++;
					fo += 1000000;  // PENALIDADE GRAVE!
				}
				
				// ============================================================
				// R5: Marca em quais dias cada disciplina tem aula
				// ============================================================
				r5[matriz.n[i][j]][i/periodos_dia]++;  // i/periodos_dia = dia
				
				// ============================================================
				// R6: Verifica compacidade (aulas adjacentes)
				// ============================================================
				fo += restricaoR6(matriz, matriz.n[i][j], i, j);
				
				// ============================================================
				// R7: Verifica capacidade da sala
				// ============================================================
				sub_r7 = disc[matriz.n[i][j]].alunos - sala[j].capacidade;
				if(sub_r7 > 0){  // Se há alunos além da capacidade
					fo += sub_r7;  // Penaliza 1 ponto por aluno extra
					restricoes_violadas[7] += sub_r7;
					// Guarda a pior violação de cada disciplina
					if(aux_mov_r7[matriz.n[i][j]][0] < sub_r7){
						aux_mov_r7[matriz.n[i][j]][0] = sub_r7;  // Guarda excesso
						aux_mov_r7[matriz.n[i][j]][1] = (j * total_periodos) + i;  // Guarda posição
					}
				}
				
				// ============================================================
				// R8: Verifica estabilidade de salas
				// ============================================================
				if(r8[matriz.n[i][j]] == -1) 
					r8[matriz.n[i][j]] = j;  // Primeira sala usada
				else if(r8[matriz.n[i][j]] != j){  // Sala diferente da primeira
					fo += 1;  // Penaliza 1 ponto
					restricoes_violadas[8]++;
					if(restricoes_violadas[8] < total_periodos)
						aux_mov_r8[restricoes_violadas[8]] = (j * total_periodos) + i;
				}

                // ============================================================
				// R9: Marca em quais dias cada professor tem aula
				// ============================================================
                if(r9[disc[matriz.n[i][j]].prof][i/periodos_dia] < 1){
				    r9[disc[matriz.n[i][j]].prof][i/periodos_dia] = 1;  // i/periodos_dia = dia
                }

				// ============================================================
				// R10: Verifica tipo da sala
				// ============================================================
				if(disc[matriz.n[i][j]].tipo_sala != sala[j].tipo_sala){
					fo += 1000000;  // Penalização grave 
					restricoes_violadas[10]++;
					// Guarda tipo correto e posição codificada
					aux_mov_r10[matriz.n[i][j]][0] = disc[matriz.n[i][j]].tipo_sala;
					aux_mov_r10[matriz.n[i][j]][1] = (j * total_periodos) + i;
				}

				// ============================================================
				// R11: Conta ocorrências da disciplina no dia
				// ============================================================
				r11[i/periodos_dia][matriz.n[i][j]]++;
				if(r11[i/periodos_dia][matriz.n[i][j]] > 1){
					aux_mov_r11[i][j] = matriz.n[i][j];  // Guarda disciplina e posição
				}
			}
		}
	}
	
	// ========================================================================
	// PÓS-PROCESSAMENTO: Calcula penalidades finais
	// ========================================================================
	
	// ============================================================
	// R2: Penaliza conflitos de professor e curso
	// ============================================================
	for(i = 0; i < total_periodos; i++){
		// Conflitos de PROFESSOR
		for(j = 0; j < professores; j++){
			if(r21[i][j] > 1){
				fo += 1000000 * (r21[i][j] - 1);  // PENALIDADE GRAVE!
				// Marca tipo de conflito (< 1000 = professor)
				if(restricoes_violadas[2] < 0) 
					restricoes_violadas[2] = 1;
				else 
					restricoes_violadas[2] += 1;
			}
		}
		// Conflitos de CURSO
		for(j = 0; j < cursos; j++){
			if(r22[i][j] > 1){
				fo += 1000000 * (r22[i][j] - 1);  // PENALIDADE GRAVE!
				// Marca tipo de conflito (>= 1000 = curso)
				if(restricoes_violadas[2] < 0) 
					restricoes_violadas[2] = 1000;
				else 
					restricoes_violadas[2] += 1000;
			}
		}
	}
	
	// ============================================================
	// R1, R5 e R9: Verifica aulas agendadas e dias mínimos
	// ============================================================
	for(i = 0; i < disciplinas; i++){
		r5_aux = 0;
		
		// R1: Verifica se número de aulas está correto
		if(r1[i] != disc[i].aulas) 
			fo += 1000000 * (modulo(r1[i], disc[i].aulas));  // PENALIDADE GRAVE!
		
		// R5: Conta em quantos dias diferentes a disciplina aparece
		for(j = 0; j < dias; j++)
			if(r5[i][j] > 0)
				r5_aux++;
		
		// Se não atingiu o mínimo de dias
		if(r5_aux < disc[i].minDias){
			fo += 5 * (disc[i].minDias - r5_aux);  // Penaliza 5 pontos por dia
			restricoes_violadas[5] += r5_aux;
			aux_mov_r5[i] = r5_aux;
		}
	}

	// R9: Conta em quantos dias diferentes o professor aparece
    for(i = 0; i < professores; i++){
        int dias_com_aula = 0;
        
        for(j = 0; j < dias; j++){
            if(r9[i][j] > 0)
                dias_com_aula++;
        }
        // Se o professor tem aulas em mais de 2 dias
        if(dias_com_aula > 2){
            fo += 5 * (dias_com_aula - 2);  // Penaliza 5 pontos por dia extra
            restricoes_violadas[9] += (dias_com_aula - 2);
            aux_mov_r9[i] = dias_com_aula;
        }
    }

	// ============================================================
	// R11: Penaliza disciplinas com múltiplas aulas no mesmo dia
	// ============================================================
	for(i = 0; i < dias; i++){
		for(j = 0; j < disciplinas; j++){
			if(r11[i][j] > 1){
				// Penaliza cada ocorrência extra
				fo += 1000000 * (r11[i][j] - 1);
				restricoes_violadas[11] += (r11[i][j] - 1);
			}
		}
	}
	
	return fo;  // Retorna valor da função objetivo
}

// ============================================================================
// MANIPULAÇÃO DE SOLUÇÕES
// ============================================================================

/*
 * CRIAMATRIZ: Aloca e inicializa uma matriz de solução
 * Todos os períodos/salas começam vazios (-1)
 */
Matriz criaMatriz(){
	Matriz matriz;
	int i, j;

	// Aloca matriz [total_periodos][salas]
	matriz.n = (int**) malloc(total_periodos * sizeof(int *));
	for(i = 0; i < total_periodos; i++){
		matriz.n[i] = (int*) malloc(salas * sizeof(int));
		// Inicializa com -1 (vazio)
		for(j = 0; j < salas; j++) 
			matriz.n[i][j] = -1;
	}
	return matriz;
}

/*
 * COPIAMATRIZ: Copia uma solução para outra
 * Copia tanto a matriz quanto o valor da função objetivo
 */
void copiaMatriz(Matriz *destino, Matriz origem){
	int i, j;
	// Copia célula por célula
	for(i = 0; i < total_periodos; i++){
		for(j = 0; j < salas; j++) 
			destino->n[i][j] = origem.n[i][j];
	}
	destino->fo = origem.fo;  // Copia FO
}

// ============================================================================
// GERAÇÃO DE VIZINHANÇA - MOVIMENTOS
// ============================================================================

/*
 * GERAVIZ: Gera uma solução vizinha através de movimentos
 * 
 * Esta é a FUNÇÃO MAIS COMPLEXA do código!
 * 
 * ESTRATÉGIA: Busca local adaptativa com múltiplos operadores
 * - Se há violações específicas, tenta corrigi-las (INTENSIFICAÇÃO)
 * - Caso contrário, faz movimentos aleatórios (DIVERSIFICAÇÃO)
 * 
 * OPERADORES DE VIZINHANÇA:
 * 1. Correção de conflitos de professor (R2)
 * 2. Correção de conflitos de curso (R2)
 * 3. Correção de compacidade (R6)
 * 4. Correção de capacidade (R7)
 * 5. Correção de estabilidade (R8)
 * 6. Troca aleatória no mesmo período
 * 7. Troca aleatória na mesma sala
 * 8. Troca totalmente aleatória
 * 
 * A escolha do operador é baseada em:
 * - Quais restrições estão sendo violadas
 * - Um sorteio aleatório ponderado
 * - A temperatura atual (T)
 */
Matriz geraViz(Matriz matriz){
	int i, j, k, l, d, busca;
	int per, sal, aux, aux2, aux3, tentativas, movimento, dia, dia_destino;

	aux = -1;
	
	// ========================================================================
	// Define número de tentativas baseado na temperatura
	// ========================================================================
	if(T < 1) tentativas = 6;
	else if(T < 10) tentativas = 5;
	else if(T < 100) tentativas = 4;
	else if(T < 1000) tentativas = 3;
	else tentativas = 2;

	// ========================================================================
	// Escolhe tipo de movimento (0-1000)
	// ========================================================================
	movimento = randomInt(0, 1000);
	
	// ========================================================================
	// MOVIMENTO 1: Correção de conflitos de PROFESSOR (R2)
	// Faixa: 0-100 + bônus proporcional ao número de conflitos
	// ========================================================================
	if((restricoes_violadas[2] != -1) && (movimento < 100 + ((restricoes_violadas[2]%1000) << 7))){
		if(restricoes_violadas[2] % 1000 > 1){
			i = 0;
			aux = -1;
			while(aux == -1){
				j = randomInt(0, disciplinas - 1);
				if(aux_mov_r21[j] != -1) 
					aux = j;
				else if(aux_mov_r21[i] != -1) 
					aux = i;
				i++;
			}

			sal = aux_mov_r21[aux] / total_periodos;
			per = aux_mov_r21[aux] - (sal * total_periodos);
			aux2 = -2;
			
			while(aux2 == -2){
				k = randomInt(0, total_periodos - 1);
				l = randomInt(0, salas - 1);
				
				if(matriz.n[k][l] == -1){
					aux2 = matriz.n[per][sal];
					matriz.n[per][sal] = -1;
					matriz.n[k][l] = aux2;
				}
				else if(r21[k][disc[matriz.n[k][l]].prof] == 0){
					aux2 = matriz.n[k][l];
					matriz.n[k][l] = matriz.n[per][sal];
					matriz.n[per][sal] = aux2;
				}
			}
			aux_mov_r21[aux] = -1;
			restricoes_violadas[2]--;
		}
		else{
			aux = randomInt(1, tentativas);
			while(aux > 0){
				i = randomInt(0, total_periodos - 1);
				j = randomInt(0, salas - 1);
				k = randomInt(0, total_periodos - 1);
				l = randomInt(0, salas - 1);
				if((matriz.n[i][j] != -1) || (matriz.n[k][l] != -1)){
					aux2 = matriz.n[i][j];
					matriz.n[i][j] = matriz.n[k][l];
					matriz.n[k][l] = aux2;
					aux--;
				}
			}
		}
	}

	// ========================================================================
	// MOVIMENTO 2: Correção de conflitos de CURSO (R2)
	// Faixa: 100 + bônus proporcional ao número de conflitos
	// ========================================================================
	else if((restricoes_violadas[2] != -1) && (movimento < 100 + (restricoes_violadas[2] >> 3))){
		if(restricoes_violadas[2] > 1000){
			i = 0;
			aux = -2;
			while(aux == -2){
				j = randomInt(0, disciplinas - 1);
				if(aux_mov_r22[j] != -1) 
					aux = j;
				else if(aux_mov_r22[i] != -1) 
					aux = i;
				i++;
			}
			aux2 = -2;

			sal = aux_mov_r22[aux] / total_periodos;
			per = aux_mov_r22[aux] - (sal * total_periodos);
			
			while(aux2 == -2){
				k = randomInt(0, total_periodos - 1);
				l = randomInt(0, salas - 1);
				
				if(matriz.n[k][l] == -1){
					aux2 = matriz.n[per][sal];
					matriz.n[per][sal] = -1;
					matriz.n[k][l] = aux2;
				}
				else if(k != per){
					aux2 = matriz.n[k][l];
					matriz.n[k][l] = matriz.n[per][sal];
					matriz.n[per][sal] = aux2;
				}
			}
			aux_mov_r22[aux] = -1;
			restricoes_violadas[2] -= 1000;
		}
		else{
			aux = randomInt(1, tentativas);
			while(aux > 0){
				i = randomInt(0, total_periodos - 1);
				j = randomInt(0, salas - 1);
				k = randomInt(0, total_periodos - 1);
				l = randomInt(0, salas - 1);
				if((matriz.n[i][j] != -1) || (matriz.n[k][l] != -1)){
					aux2 = matriz.n[i][j];
					matriz.n[i][j] = matriz.n[k][l];
					matriz.n[k][l] = aux2;
					aux--;
				}
			}
		}
	}

	// ========================================================================
	// MOVIMENTO 3: Correção de COMPACIDADE (R6)
	// Faixa: 100-200 + bônus proporcional às violações
	// ========================================================================
	else if((restricoes_violadas[6] != -1) && (movimento >= 100) && (movimento < 200 + (2 * restricoes_violadas[6]))){
		int r6_i, r6_j;
		aux = -2;
		k = 0;
		l = 0;
		
		while(aux == -2){
			i = randomInt(0, total_periodos - 1);
			j = randomInt(0, salas - 1);
			if(aux_mov_r6[i][j] != -1){
				aux = aux_mov_r6[i][j];
				r6_i = i;
				r6_j = j;
			}
			else if(aux_mov_r6[k][l] != -1){
				aux = aux_mov_r6[k][l];
				r6_i = k;
				r6_j = l;
			}
			l = (l+1) % salas;
			if(l == 0) k = (k+1) % total_periodos;
		}
		
		aux2 = -2;
		aux3 = 0;
		
		while(aux2 == -2){
			k = randomInt(0, total_periodos - 1);
			l = randomInt(0, salas - 1);
			
			if((matriz.n[k][l] == -1) && (r6_i != k)){
				matriz.n[k][l] = matriz.n[r6_i][r6_j];
				matriz.n[r6_i][r6_j] = -1;
				aux2 = 0;
			}
			else if((aux_mov_r6[k][l] != -1) && (r6_i != k)){
				aux2 = matriz.n[k][l];
				matriz.n[k][l] = matriz.n[r6_i][r6_j];
				matriz.n[r6_i][r6_j] = aux2;
				aux_mov_r6[k][l] = -1;
			}
			else if((aux3 >= tentativas) && (r6_i != k)){
				aux2 = matriz.n[k][l];
				aux = matriz.n[r6_i][r6_j];
				matriz.n[k][l] = aux;
				matriz.n[r6_i][r6_j] = aux2;
				aux_mov_r6[k][l] = -1;
			}
			else if(aux3 >= tentativas){
				aux2 = matriz.n[r6_i][r6_j];
				matriz.n[r6_i][r6_j] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			aux3++;
		}
		restricoes_violadas[6] -= 2;
		aux_mov_r6[r6_i][r6_j] = -1;
	}

	// ========================================================================
	// MOVIMENTO 4: Correção de CAPACIDADE (R7)
	// Faixa: 200-300 + bônus proporcional às violações
	// ========================================================================
	else if((restricoes_violadas[7] != -1) && (movimento >= 200) && (movimento < 300 + restricoes_violadas[7])){
		i = 0;
		j = 0;
		aux = -1;
		
		while(aux == -1){
			j = randomInt(0, disciplinas - 1);
			if(aux_mov_r7[j][0] != -1) 
				aux = j;
			else if(aux_mov_r7[i][0] != -1) 
				aux = i;
			i++;
		}
		
		aux2 = -2;
		aux3 = 0;

		sal = aux_mov_r7[aux][1] / total_periodos;
		per = aux_mov_r7[aux][1] - (sal * total_periodos);
		
		while(aux2 == -2){
			k = randomInt(0, total_periodos - 1);
			l = randomInt(0, salas - 1);
			
			if((matriz.n[k][l] == -1) && (sala[l].capacidade >= disc[aux].alunos)){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = -1;
				matriz.n[k][l] = aux2;
			}
			else if((matriz.n[k][l] != -1) && 
			        (sala[l].capacidade >= disc[aux].alunos) && 
			        (disc[matriz.n[k][l]].alunos - sala[l].capacidade > aux_mov_r7[aux][0])){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			else if((matriz.n[k][l] != -1) && (disc[matriz.n[k][l]].alunos > sala[l].capacidade)){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			else if((aux3 >= tentativas) && (sala[sal].capacidade > sala[l].capacidade)){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			else if(aux3 >= tentativas){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			aux3++;
		}
		restricoes_violadas[7]--;
		aux_mov_r7[aux][0] = -1;
		aux_mov_r7[aux][1] = -1;
	}

	// ========================================================================
	// MOVIMENTO 5: Correção de ESTABILIDADE (R8)
	// Faixa: 300-400 + bônus proporcional às violações
	// ========================================================================
	else if((restricoes_violadas[8] != -1) && (movimento >= 300) && (movimento < 400 + restricoes_violadas[8])){
		if(restricoes_violadas[8] >= total_periodos) 
			aux = randomInt(0, total_periodos - 1);
		else 
			aux = randomInt(0, restricoes_violadas[8]);
		
		sal = aux_mov_r8[aux] / total_periodos;
		per = aux_mov_r8[aux] - (sal * total_periodos);
		aux2 = -2;
		aux3 = 0;
		
		while(aux2 == -2){
			i = randomInt(0, total_periodos - 1);
			j = r8[matriz.n[per][sal]];
			
			if(matriz.n[i][j] == -1){
				matriz.n[i][j] = matriz.n[per][sal];
				matriz.n[per][sal] = -1;
				aux2 = 1;
			}
			else if(aux3 >= tentativas){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[i][j];
				matriz.n[i][j] = aux2;
			}
			aux3++;
		}
		restricoes_violadas[8]--;
		r8[aux] = -1;
	}

	// ========================================================================
	// MOVIMENTO 6: Correção de CARGA DE PROFESSORES (R9)
	// Faixa: 400-500 + bônus proporcional às violações
	// ========================================================================
	else if((restricoes_violadas[9] != -1) && (movimento >= 400) && (movimento < 500 + (restricoes_violadas[9] * 20))){
		// Encontrar professor que viola R9
		int prof_violador = -1;
		int tentativa_prof = 0;
		
		while(prof_violador == -1 && tentativa_prof < 100){
			i = randomInt(0, professores - 1);
			if(aux_mov_r9[i] > 2){  // Professor trabalha em mais de 2 dias
				prof_violador = i;
			}
			tentativa_prof++;
		}
		
		if(prof_violador != -1){
			// Identificar dias onde este professor trabalha
			int dias_trabalhados[10];
			int num_dias = 0;
			
			for(d = 0; d < dias; d++){
				if(r9[prof_violador][d] == 1){
					dias_trabalhados[num_dias++] = d;
				}
			}
			
			// Escolher dia "fonte" (para tirar aula) - escolher dia com menos aulas
			int dia_fonte = dias_trabalhados[randomInt(0, num_dias - 1)];
			
			// Escolher dia "destino" (para concentrar) - escolher outro dia onde já trabalha
			int dia_destino = dias_trabalhados[randomInt(0, num_dias - 1)];
			while(dia_destino == dia_fonte && num_dias > 1){
				dia_destino = dias_trabalhados[randomInt(0, num_dias - 1)];
			}
			
			// Encontrar uma aula deste professor no dia_fonte
			int per_fonte = -1;
			int sala_fonte = -1;
			
			for(per = dia_fonte * periodos_dia; per < (dia_fonte + 1) * periodos_dia; per++){
				for(sal = 0; sal < salas; sal++){
					if(matriz.n[per][sal] != -1 && disc[matriz.n[per][sal]].prof == prof_violador){
						per_fonte = per;
						sala_fonte = sal;
						break;
					}
				}
				if(per_fonte != -1) break;
			}
			
			if(per_fonte != -1){
				// Tentar mover para dia_destino
				aux2 = -2;
				aux3 = 0;
				
				while(aux2 == -2 && aux3 < tentativas * 2){
					k = randomInt(dia_destino * periodos_dia, (dia_destino + 1) * periodos_dia - 1);
					l = randomInt(0, salas - 1);
					
					// CASO 1: Slot vazio no dia destino
					if(matriz.n[k][l] == -1){
						matriz.n[k][l] = matriz.n[per_fonte][sala_fonte];
						matriz.n[per_fonte][sala_fonte] = -1;
						aux2 = 0;
					}
					// CASO 2: Trocar com aula de outro professor que não viola R9
					else if(aux_mov_r9[disc[matriz.n[k][l]].prof] <= 2){
						aux2 = matriz.n[k][l];
						matriz.n[k][l] = matriz.n[per_fonte][sala_fonte];
						matriz.n[per_fonte][sala_fonte] = aux2;
					}
					// CASO 3: Após tentativas, aceita qualquer troca
					else if(aux3 >= tentativas){
						aux2 = matriz.n[k][l];
						matriz.n[k][l] = matriz.n[per_fonte][sala_fonte];
						matriz.n[per_fonte][sala_fonte] = aux2;
					}
					aux3++;
				}
			}
		}
	}

	// ========================================================================
	// MOVIMENTO 7: Correção de TIPO DE SALA (R10)
	// Faixa: 500-600 + bônus proporcional às violações
	// ========================================================================
	else if((restricoes_violadas[10] != -1) && (movimento >= 500) && (movimento < 600 + restricoes_violadas[10])){
		i = 0;
		j = 0;
		aux = -1;
		
		// Procura disciplina com problema de tipo de sala
		while(aux == -1){
			j = randomInt(0, disciplinas - 1);
			if(aux_mov_r10[j][0] != -1)  
				aux = j;
			else if(aux_mov_r10[i][0] != -1) 
				aux = i;
			i++;
		}
		
		aux2 = -2;
		aux3 = 0;

		// Extrai sala e período
		sal = aux_mov_r10[aux][1] / total_periodos; 
		per = aux_mov_r10[aux][1] - (sal * total_periodos); 
		
		// Tenta alocar em sala com tipo adequado
		while(aux2 == -2){
			k = randomInt(0, total_periodos - 1);
			l = randomInt(0, salas - 1);
			
			// CASO 1: Sala vazia e com tipo correto
			if((matriz.n[k][l] == -1) && (sala[l].tipo_sala == disc[aux].tipo_sala)){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = -1;
				matriz.n[k][l] = aux2;
			}
			// CASO 2: Troca que melhora ambas 
			else if((matriz.n[k][l] != -1) && 
			        (sala[l].tipo_sala == disc[aux].tipo_sala) && 
			        (disc[matriz.n[k][l]].tipo_sala == sala[sal].tipo_sala)){ 
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			// CASO 3: Troca com disciplina também inadequada
			else if((matriz.n[k][l] != -1) && (disc[matriz.n[k][l]].tipo_sala != sala[l].tipo_sala)){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			// CASO 4: Aceita qualquer troca
			else if(aux3 >= tentativas){
				aux2 = matriz.n[per][sal];
				matriz.n[per][sal] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
			}
			aux3++;
		}
		restricoes_violadas[10]--;
		aux_mov_r10[aux][0] = -1;  
		aux_mov_r10[aux][1] = -1;
	}

	// ========================================================================
	// MOVIMENTO 8: Correção de DISTRIBUIÇÃO (R11) 
	// Faixa: 600-700 + bônus proporcional às violações
	// ========================================================================
	else if((restricoes_violadas[11] != -1) && (movimento >= 600) && (movimento < 700 + (restricoes_violadas[11] * 100))){
		// Procurar disciplina que viola R11
		int disc_r11 = -1;
		int per_r11 = -1;
		int sal_r11 = -1;
		int dia_r11 = -1;
		
		// Buscar em aux_mov_r11 
		for(per = 0; per < total_periodos && disc_r11 == -1; per++){
			for(sal = 0; sal < salas; sal++){
				if(aux_mov_r11[per][sal] != -1){ 
					disc_r11 = aux_mov_r11[per][sal];
					per_r11 = per;
					sal_r11 = sal;
					dia_r11 = per / periodos_dia;
					break;
				}
			}
		}
		
		if(disc_r11 != -1){
			aux2 = -2;
			aux3 = 0;
			
			// Tentar mover para DIA DIFERENTE
			while(aux2 == -2 && aux3 < tentativas * 3){
				k = randomInt(0, total_periodos - 1);
				l = randomInt(0, salas - 1);
				d = k / periodos_dia;
				
				// CASO 1: Dia diferente e slot vazio
				if((d != dia_r11) && (matriz.n[k][l] == -1)){
					aux2 = matriz.n[per_r11][sal_r11];
					matriz.n[per_r11][sal_r11] = -1;
					matriz.n[k][l] = aux2;
				}
				// CASO 2: Dia diferente, trocar
				else if(d != dia_r11){
					aux2 = matriz.n[k][l];
					matriz.n[k][l] = matriz.n[per_r11][sal_r11];
					matriz.n[per_r11][sal_r11] = aux2;
				}
				// CASO 3: Após tentativas, aceita qualquer mudança de dia
				else if((aux3 >= tentativas) && (d != dia_r11)){
					aux2 = matriz.n[k][l];
					matriz.n[k][l] = matriz.n[per_r11][sal_r11];
					matriz.n[per_r11][sal_r11] = aux2;
				}
				aux3++;
			}
			aux_mov_r11[per_r11][sal_r11] = -1;  
		}
		// Movimento alternativo se não encontrou
	}

	// ========================================================================
	// MOVIMENTO 9: Troca aleatória no MESMO PERÍODO
	// Faixa: 700-800
	// ========================================================================
	else if((movimento >= 700) && (movimento < 800)){
		aux = randomInt(1, tentativas << 1);
		while(aux > 0){
			i = randomInt(0, total_periodos - 1);
			j = randomInt(0, salas - 1);
			l = randomInt(0, salas - 1);
			if((matriz.n[i][j] != -1) || (matriz.n[i][l] != -1)){
				aux2 = matriz.n[i][j];
				matriz.n[i][j] = matriz.n[i][l];
				matriz.n[i][l] = aux2;
				aux--;
			}
		}
	}

	// ========================================================================
	// MOVIMENTO 10: Troca aleatória na MESMA SALA
	// Faixa: 800-900
	// ========================================================================
	else if((movimento >= 800) && (movimento < 900)){
		aux = randomInt(1, tentativas << 1);
		while(aux > 0){
			i = randomInt(0, total_periodos - 1);
			j = randomInt(0, salas - 1);
			k = randomInt(0, total_periodos - 1);
			if((matriz.n[i][j] != -1) || (matriz.n[k][j] != -1)){
				aux2 = matriz.n[i][j];
				matriz.n[i][j] = matriz.n[k][j];
				matriz.n[k][j] = aux2;
				aux--;
			}
		}
	}

	// ========================================================================
	// MOVIMENTO 11: Troca TOTALMENTE ALEATÓRIA
	// Faixa: 900-1000
	// ========================================================================
	else{
		aux = randomInt(1, tentativas);
		while(aux > 0){
			i = randomInt(0, total_periodos - 1);
			j = randomInt(0, salas - 1);
			k = randomInt(0, total_periodos - 1);
			l = randomInt(0, salas - 1);
			if((matriz.n[i][j] != -1) || (matriz.n[k][l] != -1)){
				aux2 = matriz.n[i][j];
				matriz.n[i][j] = matriz.n[k][l];
				matriz.n[k][l] = aux2;
				aux--;
			}
		}
	}
	
	return matriz;
}
// ============================================================================
// FUNÇÕES DE TEMPO E EXIBIÇÃO
// ============================================================================

/*
 * IMPRIMETEMPO: Formata e exibe tempo decorrido em HH:MM:SS.fff
 */
void imprimeTempo(float Tempo, int hora, int minuto){
	if(hora < 10)
		if(minuto < 10)
			if(Tempo < 10)
				printf("\nTempo: 0%d:0%d:0%.3fs", hora, minuto, Tempo);
			else printf("\nTempo: 0%d:0%d:%.3fs", hora, minuto, Tempo);
		else if(Tempo < 10)
				printf("\nTempo: 0%d:%d:0%.3fs", hora, minuto, Tempo);
			else printf("\nTempo: 0%d:%d:%.3fs", hora, minuto, Tempo);
	else if(minuto < 10)
			if(Tempo < 10)
				printf("\nTempo: %d:0%d:0%.3fs", hora, minuto, Tempo);
			else printf("\nTempo: %d:0%d:%.3fs", hora, minuto, Tempo);
		else if(Tempo < 10)
				printf("\nTempo: %d:%d:0%.3fs", hora, minuto, Tempo);
			else printf("\nTempo: %d:%d:%.3fs", hora, minuto, Tempo);
}

/*
 * ALTERA_PARAMETROS: Função placeholder para ajuste dinâmico de parâmetros
 */
void altera_parametros(){
	// Não implementada
}

// ============================================================================
// ALGORITMO SIMULATED ANNEALING
// ============================================================================

/*
 * SA: Implementação do Simulated Annealing
 * 
 * CONCEITO: Metaheurística inspirada no processo de recozimento de metais
 * 
 * FUNCIONAMENTO:
 * 1. Começa com temperatura alta (T_inicial)
 * 2. A cada iteração:
 *    - Gera solução vizinha
 *    - Se melhor: aceita
 *    - Se pior: aceita com probabilidade exp(-delta/T)
 * 3. Reduz temperatura (T = T * alpha)
 * 4. Repete até T < T_final ou solução ótima
 * 
 * PARÂMETROS:
 * - T_inicial: Temperatura inicial (exploração)
 * - T_final: Temperatura final (refinamento)
 * - alpha: Taxa de resfriamento (0.93-0.995)
 * - maxIteracoes: Iterações por temperatura
 * 
 * TÉCNICAS ESPECIAIS:
 * - Reaquecimento: Aumenta T quando estagnado
 * - Ajuste dinâmico: Varia parâmetros conforme T
 * - Amplificação delta: Multiplica diferença por 4
 */
Matriz SA(Matriz inicial){
	// Aloca estruturas para soluções
	Matriz atual = criaMatriz();
	Matriz melhor = criaMatriz();
	Matriz viz = criaMatriz();

	clock_t inicio, fim;  // Para medir tempo

	float Tempo, Temp_reaquecimento;
	int i, delta, hora = 0, minuto = 0;
	int reaquecimento = 1;       // Contador de reaquecimentos
	int fim_forcado = 0;         // Contador de iterações sem melhora

	// ========================================================================
	// INICIALIZAÇÃO DOS PARÂMETROS
	// ========================================================================
	
	Tinicial = 1000000;          // Temperatura inicial muito alta
	Tfinal = 0.00001;            // Temperatura final muito baixa
	Temp_reaquecimento = Tfinal * 10;  // Limiar para reaquecimento

	inicio = clock();            // Marca tempo inicial

	copiaMatriz(&atual, inicial);    // Copia solução inicial
	copiaMatriz(&melhor, atual);     // Melhor = inicial

	T = Tinicial;                // Começa na temperatura inicial

	// ========================================================================
	// LOOP PRINCIPAL DO SIMULATED ANNEALING
	// ========================================================================
	
	while ((T > Tfinal) && (melhor.fo > 0) && (fim_forcado < 8000)){
		fim_forcado++;  // Incrementa contador de estagnação
		
		// ====================================================================
		// AJUSTE DINÂMICO DE PARÂMETROS baseado na temperatura
		// ====================================================================
		
		if(T > 1000){
			// Temperatura MUITO ALTA: Exploração rápida
			maxIteracoes = 600;
			alpha = 0.98;    // Resfriamento mais rápido
		}
		else if(T > 100){
			// Temperatura ALTA: Exploração moderada
			maxIteracoes = 800;
			alpha = 0.97;
		}
		else if(T > 10){
			// Temperatura MÉDIA: Balanceamento
			maxIteracoes = 1000;
			alpha = 0.98;
		}
		else if(T > 1){
			// Temperatura BAIXA: Intensificação
			maxIteracoes = 1200;
			alpha = 0.99;
		}
		else if(T > 0.1){
			// Temperatura MUITO BAIXA: Refinamento
			maxIteracoes = 1500;
			alpha = 0.993;
		}
		else{
			// Temperatura EXTREMAMENTE BAIXA: Busca local
			maxIteracoes = 1200;
			alpha = 0.995;  // Resfriamento muito lento
		}
		
		// ====================================================================
		// ITERAÇÕES NA TEMPERATURA ATUAL
		// ====================================================================
		
		for(i = 0; i < maxIteracoes; i++){
			// Copia solução atual para gerar vizinho
			copiaMatriz(&viz, atual);
			
			// Recalcula FO apenas em temperatura baixa (refinamento)
			if(T < 100)
				viz.fo = calcula_FO(viz);
			
			// Gera solução vizinha
			viz = geraViz(viz);
			
			// Calcula FO da vizinha
			viz.fo = calcula_FO(viz);
			
			// Calcula diferença (delta)
			delta = viz.fo - atual.fo;
			delta = delta << 2;  // Multiplica por 4 (amplifica diferença)

			// ================================================================
			// CRITÉRIO DE ACEITAÇÃO
			// ================================================================
			
			if(delta < 0){
				// CASO 1: Vizinho é MELHOR - sempre aceita
				copiaMatriz(&atual, viz);
				
				// Se é o melhor global
				if(atual.fo < melhor.fo){
					copiaMatriz(&melhor, atual);
					fim_forcado = 0;  // Reseta contador de estagnação
					
					// Guarda no histórico
					aux_mat = (aux_mat + 1) % HISTORICO;
					mat_solucao_tempo[aux_mat][0] = melhor.fo;
					mat_solucao_tempo[aux_mat][1] = (clock() - inicio) / 1000;
					
					// Exibe progresso
					imprimeTempo(Tempo, hora, minuto);
					if(viz.fo >= 1000000)
						printf("|  Temp(K) = %.6f \t|  atual.fo = %d \t|  viz.fo = %d\t|  melhor.fo = %d\t (%d)(%d)(%d)", 
						       T, atual.fo, viz.fo, melhor.fo, programa, rotina, fim_forcado);
					else 
						printf("|  Temp(K) = %.4f \t|  atual.fo = %d \t|  viz.fo = %d   \t|  melhor.fo = %d\t (%d)(%d)(%d)", 
						       T, atual.fo, viz.fo, melhor.fo, programa, rotina, fim_forcado);
				}
			}
			// CASO 2: Vizinho é PIOR - aceita com CRITÉRIO DE METROPOLIS
			else if(randomDouble(0.0, 1.0) < (exp(-1 * (delta / T)))) {
				copiaMatriz(&atual, viz);
			}
			// CASO 3: Vizinho é PIOR e não passou no critério - rejeita
		}

		// ====================================================================
		// ATUALIZAÇÃO DO TEMPO
		// ====================================================================
		
		fim = clock();
		Tempo = (double)(fim - inicio) / 1000;
		Tempo -= ((hora * 3600) + (minuto * 60));
		if(Tempo >= 60){
			minuto++;
			if(minuto == 60){
				minuto -= 60;
				hora++;
			}
		}

		// ====================================================================
		// REAQUECIMENTO (Diversificação)
		// ====================================================================
		
		if((T < Temp_reaquecimento) && (reaquecimento > 0)){
			T = Tinicial * 0.1;  // Reaquecer para 10% da temperatura inicial
			reaquecimento--;     // Só reaquecer uma vez
		}
		else {
			T *= alpha;  // Resfriamento normal
		}
	}
	
	// ========================================================================
	// FINALIZAÇÃO
	// ========================================================================
	
	printf("\nT = %.6f, Tfinal = %f, melhor.fo = %d", T, Tfinal, melhor.fo);

	printf("\e[H\e[2J");  // Limpa terminal
	printf("\n");
	imprimeTempo(Tempo, hora, minuto);
	printf("\t| Temp(K) = %.4f \t| FO = %d \t| Melhor FO = %d", T, atual.fo, melhor.fo);
	printf("\nSimulação concluída. Digite o nome do arquivo para gravar os dados.");
	printf("\n\t -> ");

	return melhor;  // Retorna melhor solução encontrada
}

// ============================================================================
// GERAÇÃO DE SOLUÇÃO INICIAL
// ============================================================================

/*
 * SOLUCAOINICIAL: Gera solução inicial através de heurística construtiva
 * 
 * ESTRATÉGIA: Alocação aleatória com preferência por respeitcar R4 e R7
 * 
 * FUNCIONAMENTO:
 * 1. Para cada disciplina:
 *    a. Tenta alocar todas as aulas
 *    b. Prefere: posição vazia, sala com capacidade, período disponível
 *    c. Se não consegue, força alocação em qualquer lugar
 * 
 * RESULTADO: Solução possivelmente INVIÁVEL (com violações)
 * O SA vai melhorar esta solução
 */
Matriz solucaoInicial(){
	int i, j, k, cont, atribuicoes;
	Matriz matriz = criaMatriz();

	// Para cada disciplina
	for(j = 0; j < disciplinas; j++){
		atribuicoes = disc[j].aulas;  // Número de aulas a alocar
		printf("disc[%d].aulas: %d\n", j, disc[j].aulas);
		cont = 0;
		
		// Enquanto há aulas para alocar
		while(atribuicoes > 0){
			i = randomInt(0, total_periodos - 1);  // Período aleatório
			k = randomInt(0, salas - 1);            // Sala aleatória
			
			// TENTATIVA 1: Posição vazia, capacidade OK, tipo de sala OK, sem restrição R4
			if((matriz.n[i][k] == -1) && 
			   (sala[k].capacidade >= disc[j].alunos) && 
			   (restricaoR4(j, i) == 0) && (sala[k].tipo_sala>= disc[j].tipo_sala)){
				matriz.n[i][k] = j;  // Aloca
				atribuicoes--;
				cont -= 3;           // Reinicia contador
			}
			else {
				cont++;  // Incrementa contador de falhas
			}
			
			// TENTATIVA 2: Após muitas falhas, força alocação
			if(cont > 2){
				if(matriz.n[i][k] == -1){
					matriz.n[i][k] = j;  // Aloca forçadamente
					atribuicoes--;
					cont -= 3;
				}
			}
		}
	}
	
	// Calcula FO da solução inicial
	matriz.fo = calcula_FO(matriz);
	imprimeSolucao(matriz);
	return matriz;
}

// ============================================================================
// SALVAMENTO DE RESULTADOS
// ============================================================================

/*
 * SALVARESULTADO: Salva solução e estatísticas em arquivo
 * 
 * Salva:
 * - Dados da instância
 * - Tempo de execução
 * - Função Objetivo
 * - Relatório de violações
 * - Solução final (grade)
 * - Histórico de melhorias
 */
void salvaResultado(Matriz matriz, char *str){
	char res[SAIDA];
	strcpy(res, "");

	FILE *fp = 0;
	fp = fopen(str, "w");  // Primeira execução: cria arquivo
	
	if(!fp){
		printf("ERRO! - Não foi possível salvar os dados.\n");
		return;
	}
	
	int i, j, k, p, a;

	// ========================================================================
	// PRIMEIRA EXECUÇÃO: Salva dados da instância
	// ========================================================================
	
	if(rotina >= 0){
		// Informações básicas
		sprintf(res, "Nome: %s\n", nome);
		sprintf(res, "%sDisciplinas: %d\n", res, disciplinas);
		sprintf(res, "%sProfessores: %d\n", res, professores);
		sprintf(res, "%sSalas: %d\n", res, salas);
		sprintf(res, "%sDias: %d\n", res, dias);
		sprintf(res, "%sPeriodos por dia: %d\n", res, periodos_dia);
		sprintf(res, "%sCursos: %d\n", res, cursos);
		sprintf(res, "%sRestricoes: %d\n", res, restricoes);
		
		// Valor da Função Objetivo
		sprintf(res, "%sFunção Objetivo (FO): %d\n", res, matriz.fo);
		
		// Relatório de violações
		sprintf(res, "%s\n============ RELATÓRIO DE VIOLAÇÕES ============\n", res);
		sprintf(res, "%sR1 (Aulas incorretas):        %d\n", res, restricoes_violadas[1] > 0 ? restricoes_violadas[1] : 0);
		sprintf(res, "%sR2 (Conflitos prof/curso):    %d (prof: %d, curso: %d)\n", res,
		       restricoes_violadas[2] > 0 ? restricoes_violadas[2] : 0,
		       restricoes_violadas[2] > 0 ? restricoes_violadas[2] % 1000 : 0,
		       restricoes_violadas[2] > 0 ? restricoes_violadas[2] / 1000 : 0);
		sprintf(res, "%sR4 (Indisponibilidade):       %d\n", res, restricoes_violadas[4] > 0 ? restricoes_violadas[4] : 0);
		sprintf(res, "%sR5 (Dias mínimos):            %d\n", res, restricoes_violadas[5] > 0 ? restricoes_violadas[5] : 0);
		sprintf(res, "%sR6 (Compacidade):             %d\n", res, restricoes_violadas[6] > 0 ? restricoes_violadas[6] : 0);
		sprintf(res, "%sR7 (Capacidade sala):         %d\n", res, restricoes_violadas[7] > 0 ? restricoes_violadas[7] : 0);
		sprintf(res, "%sR8 (Estabilidade sala):       %d\n", res, restricoes_violadas[8] > 0 ? restricoes_violadas[8] : 0);
		sprintf(res, "%sR9 (Prof max 2 dias):         %d\n", res, restricoes_violadas[9] > 0 ? restricoes_violadas[9] : 0);
		sprintf(res, "%sR10 (Tipo de sala):           %d\n", res, restricoes_violadas[10] > 0 ? restricoes_violadas[10] : 0);
		sprintf(res, "%sR11 (Disciplina no mesmo dia):%d\n", res, restricoes_violadas[11] > 0 ? restricoes_violadas[11] : 0);
		sprintf(res, "%s=================================================\n", res);

		for(a = 0; res[a]; a++)	
			putc(res[a], fp);
		strcpy(res, "");
		
		// Lista de disciplinas
		sprintf(res, "\n\nDisciplinas:");
		for(a = 0; res[a]; a++) 
			putc(res[a], fp);
		strcpy(res, "");
		for(i = 0; i < disciplinas; i++){
			sprintf(res, "%s\nDscpl: %s |Prof: %s\t|Aulas: %d\t|MinDias: %d\t|Alunos: %d\t|TipoSala: %d",
        			res, disc[i].nome, disc[i].profe, disc[i].aulas, disc[i].minDias, disc[i].alunos, disc[i].tipo_sala);
			for(a = 0; res[a]; a++) 
				putc(res[a], fp);
			strcpy(res, "");
		}
		
		// Lista de salas
		sprintf(res, "\n\nSalas:");
		for(a = 0; res[a]; a++) 
			putc(res[a], fp);
		strcpy(res, "");
		for(i = 0; i < salas; i++){
			sprintf(res, "%s\nSala: %s\t|Capacidade: %d\t|TipoSala: %d", 
        			res, sala[i].nome, sala[i].capacidade, sala[i].tipo_sala);
			for(a = 0; res[a]; a++) 
				putc(res[a], fp);
			strcpy(res, "");
		}
		
		// Lista de cursos
		sprintf(res, "\n\nCursos:");
		for(a = 0; res[a]; a++) 
			putc(res[a], fp);
		strcpy(res, "");
		for(i = 0; i < cursos; i++){
			sprintf(res, "%s\nCurso: %s\t|# Dspl: %d", res, curso[i].nome, curso[i].qtDisc);
			for(a = 0; res[a]; a++) 
				putc(res[a], fp);
			strcpy(res, "");
			for(p = 0; p < curso[i].qtDisc; p++){
				sprintf(res, "%s |%s\t", res, disc[curso[i].disciplina[p]].nome);
				for(a = 0; res[a]; a++) 
					putc(res[a], fp);
				strcpy(res, "");
			}
		}
		
		// Grade (tabela período x sala)
		sprintf(res, "[Dia/Per");
		for(a = 0; res[a]; a++) 
			putc(res[a], fp);
		for(j = 0; j < salas; j++){
			sprintf(res, "|%s\t", sala[j].nome);
			for(a = 0; res[a]; a++) 
				putc(res[a], fp);
		}

		sprintf(res, "|]\n");
		for(i = 0; i < total_periodos; i++){
			sprintf(res, "%s[ %d, %d\t", res, i/periodos_dia, i%periodos_dia);
			for(j = 0; j < salas; j++){
				if(matriz.n[i][j] < 0) 
					sprintf(res, "%s|-----\t", res);
				else 
					sprintf(res, "%s|%s\t", res, disc[matriz.n[i][j]].nome);
			}
			sprintf(res, "%s|]\n", res);
			for(a = 0; res[a]; a++) 
				putc(res[a], fp);
			strcpy(res, "");
		}
	}

	// ========================================================================
	// HISTÓRICO DE BUSCA
	// ========================================================================
	
	if(rotina >= 0)
		sprintf(res, "\n\nHistorico de busca (tempo em segundos e valor da FO):");
	for(a = 0; res[a]; a++) 
		putc(res[a], fp);
	
	for(i = 0; i < HISTORICO; i++){
		if(i == 0){
			sprintf(res, "\n\n\n%dº Execução: ***FO = %d***\n", rotina+1, matriz.fo);
			for(a = 0; res[a]; a++) 
				putc(res[a], fp);
		}
		j = mat_solucao_tempo[aux_mat][0];  // FO
		k = mat_solucao_tempo[aux_mat][1];  // Tempo

		sprintf(res, "\n%dº: %d  %d", i+1, k, j);
		for(a = 0; res[a]; a++) 
			putc(res[a], fp);
		
		aux_mat--;
		if(aux_mat < 0)
			aux_mat = HISTORICO - 1;
	}

	fclose(fp);
}

// ============================================================================
// FUNÇÃO PRINCIPAL
// ============================================================================

/*
 * MAIN: Função principal do programa
 * 
 * FUNCIONAMENTO:
 * 1. Inicializa parâmetros
 * 2. Para cada instância:
 *    a. Lê arquivo
 *    b. Gera solução inicial
 *    c. Aplica Simulated Annealing
 *    d. Salva resultados
 * 3. Libera memória
 */

void imprimeViolacoes(){
    int i, j, total;
    int profs_violando = 0;
    
    printf("\n============ RELATÓRIO DE VIOLAÇÕES ============\n");
    printf("R1 (Aulas incorretas):        %d\n", restricoes_violadas[1] > 0 ? restricoes_violadas[1] : 0);
    printf("R2 (Conflitos prof/curso):    %d (prof: %d, curso: %d)\n", 
           restricoes_violadas[2] > 0 ? restricoes_violadas[2] : 0,
           restricoes_violadas[2] > 0 ? restricoes_violadas[2] % 1000 : 0,
           restricoes_violadas[2] > 0 ? restricoes_violadas[2] / 1000 : 0);
    printf("R4 (Indisponibilidade):       %d\n", restricoes_violadas[4] > 0 ? restricoes_violadas[4] : 0);
    printf("R5 (Dias mínimos):            %d\n", restricoes_violadas[5] > 0 ? restricoes_violadas[5] : 0);
    printf("R6 (Compacidade):             %d\n", restricoes_violadas[6] > 0 ? restricoes_violadas[6] : 0);
    printf("R7 (Capacidade sala):         %d\n", restricoes_violadas[7] > 0 ? restricoes_violadas[7] : 0);
    printf("R8 (Estabilidade sala):       %d\n", restricoes_violadas[8] > 0 ? restricoes_violadas[8] : 0);
    printf("R9 (Prof max 2 dias):         %d\n", restricoes_violadas[9] > 0 ? restricoes_violadas[9] : 0);
    printf("R10 (Tipo de sala):           %d\n", restricoes_violadas[10] > 0 ? restricoes_violadas[10] : 0);
    printf("R11 (Disciplina no mesmo dia):%d\n", restricoes_violadas[11] > 0 ? restricoes_violadas[11] : 0); 
    printf("=================================================\n");
    
}


// ============================================================================
// FUNÇÃO AUXILIAR: EXTRAIR DIAS DA MATRIZ
// ============================================================================

int** extraiDiasDaMatriz(Matriz matriz_integral, int num_profs, int num_dias, int periodos_total, int periodos_por_dia){
    int i, j, dis, prof, dia;
    
    // Aloca matriz de dias [professores][dias]
    int** dias_ocupados = (int**) malloc(num_profs * sizeof(int*));
    for(i = 0; i < num_profs; i++){
        dias_ocupados[i] = (int*) malloc(num_dias * sizeof(int));
        // Inicializa com 0 (não trabalha neste dia)
        for(j = 0; j < num_dias; j++){
            dias_ocupados[i][j] = 0;
        }
    }
    
    // Percorre toda a matriz da grade integral
    for(i = 0; i < periodos_total; i++){
        for(j = 0; j < salas; j++){
            dis = matriz_integral.n[i][j];  // Disciplina alocada
            
            if(dis != -1){  // Se há disciplina alocada
                prof = disc[dis].prof;      // ID do professor
                dia = i / periodos_por_dia; // Extrai o dia do período
                
                // Segurança: verifica limites
                if(prof < num_profs && dia < num_dias){
                    dias_ocupados[prof][dia] = 1;  // Marca que professor trabalha neste dia
                }
            }
        }
    }
    
    return dias_ocupados;
}

Matriz construcao(char *arquivo_entrada, char *arquivo_saida, int **dias_integral, int usar_integral){

	Matriz matriz;
	int i;

	T = 10000;
	execucao = 0;

	printf("\e[H\e[2J");  // Limpa terminal
	programa = 1;
	
	// Inicializa contadores
	professores = 0;
	disciplinas = 0;
	salas = 0;
	dias = 0;
	periodos_dia = 0;
	cursos = 0;
	restricoes = 0;

	if(rotina == 0){
		num_exec = 1;  // Número de execuções
	}
	
	usar_restricao_integral = usar_integral;

	if(usar_integral && dias_integral != NULL){
		dias_ocupados_integral = dias_integral;
	}
	
	
	if(!leArquivos(arquivo_entrada)){
		printf("\n\nERRO! - Houve um problema para ler o arquivo. Tente novamente\n\n");
		matriz.fo = -1;
		return matriz;
	}
	
	printf("\e[H\e[2J");
	
	// Gera solução inicial
	matriz = solucaoInicial();
	
	// Aplica Simulated Annealing
	matriz = SA(matriz);
	
	// Recalcula FO final
	calcula_FO(matriz);
	
	// Exibe solução
	imprimeSolucao(matriz);

	// Exibe relatório de violações
	imprimeViolacoes();
	
	
	salvaResultado(matriz, arquivo_saida);
	
	printf("\nArquivo %s com as informações criado com sucesso.\n", nome);
	rotina++;

	return matriz;
}

int main(){
    Matriz integral, noturno;
    int** dias_integral = NULL;
    
    int num_dias_integral, periodos_total_integral, periodos_por_dia_integral;
    
    // ========================================================================
    // GRADE 1: INTEGRAL
    // ========================================================================
    
    integral = construcao("instUnifesp_integral", "resultados/instUnifesp_integral7", NULL, 0);
    
    if(integral.fo == -1){
        return 1;
    }
    
    int num_profs_salvo = professores;
    int num_salas_salvo = salas;
    int num_dias_salvo = dias;
    int periodos_total_salvo = total_periodos;
    int periodos_por_dia_salvo = periodos_dia;

    
     dias_integral = extraiDiasDaMatriz(
        integral, 
        num_profs_salvo, 
        num_dias_salvo,
        periodos_total_salvo, 
        periodos_por_dia_salvo
    );
 
	num_profs_da_integral = num_profs_salvo;
    
    // ========================================================================
    // GRADE 2: NOTURNA
    // ========================================================================
    
	rotina = 1;
    noturno = construcao("instUnifesp_noturno", "resultados/instUnifesp_noturno7", dias_integral, 1);
    
    if(noturno.fo == -1){
        for(int i = 0; i < num_profs_da_integral; i++){
            free(dias_integral[i]);
        }
        free(dias_integral);
        return 1;
    }
    
    // ========================================================================
    // LIBERAÇÃO FINAL COMPLETA
    // ========================================================================
    
    // Libera matriz de dias
    for(int i = 0; i < num_profs_da_integral; i++){
        free(dias_integral[i]);
    }
    free(dias_integral);
    
    // Libera matrizes de controle bidimensionais
    for(int i = 0; i < total_periodos; i++){
        free(r21[i]);
        free(r22[i]);
        free(aux_mov_r6[i]);
        free(aux_mov_r11[i]);
    }
    
    for(int i = 0; i < disciplinas; i++){
        free(r5[i]);
        free(aux_mov_r7[i]);
        free(aux_mov_r10[i]);
        free(disc[i].cursos);
    }
    
    for(int i = 0; i < professores; i++){
        free(r9[i]);
    }
    
    for(int i = 0; i < dias; i++){
        free(r11[i]);
    }
    
    // Libera ponteiros principais
    free(r21);
    free(r22);
    free(aux_mov_r6);
    free(aux_mov_r11);
    free(r5);
    free(r9);
    free(r11);
    free(aux_mov_r7);
    free(aux_mov_r10);
    
    // Libera estruturas auxiliares
    free(aux_mov_r21);
    free(aux_mov_r22);
    free(aux_mov_r4);
    free(aux_mov_r5);
    free(aux_mov_r8);
    free(aux_mov_r9);
    free(r1);
    free(r8);
    
    // Libera dados do problema
    free(disc);
    
    for(int i = 0; i < cursos; i++){
        free(curso[i].disciplina);
    }
    free(curso);
    free(prof);
    free(sala);
    free(restricao);
    free(posicao_restricao[0]);
    free(posicao_restricao[1]);
    
    printf("\n\n✓ Todas as grades foram criadas com sucesso!\n\n");
    
    return 0;
}


/*
 * ============================================================================
 * FIM DO CÓDIGO
 * ============================================================================
 * 
 * RESUMO DO ALGORITMO:
 * 
 * 1. REPRESENTAÇÃO: Matriz [períodos x salas]
 * 2. FUNÇÃO OBJETIVO: Soma ponderada de violações
 * 3. SOLUÇÃO INICIAL: Heurística construtiva aleatória
 * 4. BUSCA LOCAL: Simulated Annealing com 8 operadores de vizinhança
 * 5. ESTRATÉGIA: Busca adaptativa direcionada por violações
 * 
 * CONCEITOS DE PO APLICADOS:
 * - Programação Inteira com restrições
 * - Função de penalização
 * - Metaheurística (SA)
 * - Busca local estocástica
 * - Critério de Metropolis
 * - Estruturas de vizinhança múltiplas
 * - Reaquecimento (diversificação)
 * - Ajuste dinâmico de parâmetros
 * 
 * ============================================================================
 */