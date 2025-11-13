# Gambiarra Arena - Orquestração com LangChain

Este documento explica como usar o sistema de orquestração avançada com LangChain para melhorar significativamente a qualidade das respostas na arena.

## Arquivos

- **`orchestration.py`** - Módulo principal com classes de orquestração LangChain
- **`gambiarra-arena-client-langchain.py`** - Cliente da arena integrado com LangChain
- **`gambiarra-arena-client.py`** - Cliente básico sem LangChain (original)

## Instalação

```bash
pip install -r requeriments.txt
```

## Componentes

### 1. ArenaOrchestrator (Básico)

Orquestrador básico que usa LangChain para processar prompts com templates sofisticados.

**Características:**
- Templates de prompt otimizados
- Streaming de tokens em tempo real
- Configuração flexível de temperatura e max_tokens

### 2. EnhancedArenaOrchestrator (Avançado)

Versão aprimorada com estratégias customizáveis.

**Estratégias disponíveis:**
- `accuracy_focused` - Foco em precisão (temperatura 0.5, 500 tokens)
- `speed_focused` - Foco em velocidade (temperatura 0.3, 200 tokens)
- `detailed` - Respostas detalhadas (temperatura 0.8, 1000 tokens)

## Configuração

Edite as variáveis no topo de `gambiarra-arena-client-langchain.py`:

```python
# Conexão
HOST = "192.168.0.212"  # IP do servidor
PIN = "937414"  # PIN da sessão
PARTICIPANT_ID = "meuId"
NICKNAME = "Python LangChain"

# Ollama
OLLAMA_HOST = "localhost"
OLLAMA_MODEL = "qwen3:0.6b"

# LangChain
USE_ENHANCED = True  # Usar versão aprimorada
TEMPERATURE = 0.7  # Criatividade (0.0-1.0)
MAX_TOKENS = 500  # Limite de tokens
STRATEGY = "accuracy_focused"  # Estratégia (se USE_ENHANCED=True)
```

## Uso

### Cliente com LangChain (Recomendado)

```bash
python gambiarra-arena-client-langchain.py
```

### Cliente Básico

```bash
python gambiarra-arena-client.py
```

## Vantagens do LangChain

### 1. Templates Sofisticados
O sistema usa templates de prompt profissionais que:
- Definem o papel do assistente
- Fornecem diretrizes claras
- Adicionam contexto relevante
- Estruturam a resposta esperada

### 2. Estratégias Adaptativas
Com o `EnhancedArenaOrchestrator`, você pode:
- Ajustar o comportamento baseado no tipo de pergunta
- Otimizar para velocidade ou qualidade
- Controlar o nível de detalhe das respostas

### 3. Streaming Eficiente
- Tokens são enviados em tempo real
- Latência do primeiro token minimizada
- Métricas precisas de performance

### 4. Extensibilidade
O módulo `orchestration.py` pode ser facilmente estendido com:
- Múltiplas chains em sequência
- Memory para contexto entre rounds
- Agents para tarefas complexas
- Ferramentas customizadas

## Exemplos de Uso Programático

### Exemplo 1: Uso Básico

```python
from orchestration import create_orchestrator

# Criar orquestrador
orchestrator = create_orchestrator(
    ollama_model="llama3.2",
    temperature=0.7
)

# Processar prompt com streaming
tokens, response = await orchestrator.process_prompt_streaming(
    user_prompt="What is machine learning?",
    context="Technical explanation"
)

print(f"Total tokens: {len(tokens)}")
print(f"Response: {response}")
```

### Exemplo 2: Uso Avançado com Estratégias

```python
from orchestration import create_orchestrator

# Criar orquestrador aprimorado
orchestrator = create_orchestrator(
    ollama_model="mistral",
    enhanced=True,
    temperature=0.8,
    max_tokens=1000
)

# Criar chain com estratégia específica
chain = orchestrator.create_enhanced_chain(
    strategy="detailed"
)

# Executar
response = chain.run(
    user_input="Explain quantum computing",
    context="Scientific explanation for experts",
    strategy="detailed"
)
```

### Exemplo 3: Integração Manual

```python
import asyncio
from orchestration import ArenaOrchestrator, TokenCollectorCallback

async def custom_processing():
    orchestrator = ArenaOrchestrator(
        ollama_host="localhost",
        ollama_model="llama3.2"
    )

    callback = TokenCollectorCallback()
    chain = orchestrator.create_chain(callback_handler=callback)

    # Executar em background
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(
        None,
        lambda: chain.run(
            user_input="Your question here",
            context="Custom context"
        )
    )

    # Processar tokens conforme chegam
    while True:
        token = await callback.token_queue.get()
        if token is None:
            break
        print(token, end='', flush=True)

    await task
```

## Personalização

### Modificar Templates

Edite `orchestration.py` para customizar os templates:

```python
self.prompt_template = PromptTemplate(
    input_variables=["user_input", "context"],
    template="""Seu template customizado aqui...

Context: {context}
Question: {user_input}

Answer:"""
)
```

### Adicionar Novas Estratégias

No método `create_enhanced_chain`:

```python
if strategy == "sua_estrategia":
    temp = 0.6
    max_tok = 300
```

### Usar Memory (Context entre rounds)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)
```

## Troubleshooting

### Erro: "Import langchain could not be resolved"

Execute:
```bash
pip install langchain langchain-community
```

### Ollama não responde

1. Verifique se o Ollama está rodando: `ollama list`
2. Teste o modelo: `ollama run llama3.2`
3. Verifique a URL: `http://localhost:11434`

### Tokens não estão sendo enviados

O callback pode não estar funcionando. Verifique:
- Se `callbacks` foi passado para o LLM
- Se o modelo suporta streaming
- Logs de erro no console

## Performance

Comparação de estratégias (aproximado):

| Estratégia | Velocidade | Qualidade | Tokens | Use quando |
|-----------|-----------|-----------|--------|-----------|
| speed_focused | ⚡⚡⚡ | ⭐⭐ | 200 | Tempo é crítico |
| accuracy_focused | ⚡⚡ | ⭐⭐⭐ | 500 | Equilíbrio |
| detailed | ⚡ | ⭐⭐⭐⭐ | 1000 | Qualidade máxima |

## Próximos Passos

- [ ] Adicionar suporte a RAG (Retrieval Augmented Generation)
- [ ] Implementar agents com ferramentas
- [ ] Adicionar memory persistente entre sessões
- [ ] Criar chains especializadas por tipo de pergunta
- [ ] Implementar ensemble de múltiplos modelos

## Recursos

- [LangChain Docs](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
