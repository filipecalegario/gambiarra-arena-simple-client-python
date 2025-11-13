#!/usr/bin/env python3
"""
Módulo de orquestração usando LangChain para processar prompts
de forma mais sofisticada e incrementada.
"""
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, List, Optional
import queue
import threading
import asyncio


class TokenCollectorCallback(BaseCallbackHandler):
    """
    Callback customizado para coletar tokens durante o streaming
    e enviá-los para o servidor da arena.
    Thread-safe usando queue.Queue ao invés de asyncio.Queue.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.token_queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Chamado quando um novo token é gerado"""
        with self.lock:
            self.tokens.append(token)
        # Adiciona o token na fila de forma thread-safe
        self.token_queue.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        """Chamado quando a geração termina"""
        self.token_queue.put(None)  # Sentinel value


class ArenaOrchestrator:
    """
    Orquestrador principal que usa LangChain para processar prompts
    de forma mais sofisticada.
    """

    def __init__(
        self,
        ollama_host: str = "localhost",
        ollama_model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Inicializa o orquestrador com configurações do modelo.

        Args:
            ollama_host: Host do servidor Ollama
            ollama_model: Nome do modelo a ser usado
            temperature: Temperatura para geração (0.0 a 1.0)
            max_tokens: Número máximo de tokens a gerar
        """
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Template de prompt sofisticado
        self.prompt_template = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""You are an expert AI assistant participating in a competitive arena.
Your goal is to provide the best possible answer to demonstrate your capabilities.

Context: {context}

Guidelines:
- Be accurate, informative, and comprehensive
- Provide well-structured responses
- Use clear and engaging language
- Stay focused on the question
- Demonstrate deep understanding

User Question: {user_input}

Your Response:"""
        )

    def create_chain(self, callback_handler: Optional[BaseCallbackHandler] = None):
        """
        Cria uma chain do LangChain com o modelo e callbacks configurados.

        Args:
            callback_handler: Callback customizado para processar tokens

        Returns:
            LLMChain configurada
        """
        callbacks = []
        if callback_handler:
            callbacks.append(callback_handler)

        # Configurar LLM do Ollama
        llm = Ollama(
            base_url=f"http://{self.ollama_host}:11434",
            model=self.ollama_model,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            callbacks=callbacks,
        )

        # Criar chain
        chain = LLMChain(
            llm=llm,
            prompt=self.prompt_template,
            verbose=False,
        )

        return chain

    async def process_prompt_streaming(
        self,
        user_prompt: str,
        context: str = "General knowledge task"
    ) -> tuple[List[str], str]:
        """
        Processa um prompt com streaming de tokens.

        Args:
            user_prompt: Prompt do usuário
            context: Contexto adicional para o prompt

        Returns:
            Tupla com (lista de tokens, resposta completa)
        """
        callback = TokenCollectorCallback()
        chain = self.create_chain(callback_handler=callback)

        # Executar chain em uma thread separada
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            None,
            lambda: chain.run(user_input=user_prompt, context=context)
        )

        # Coletar tokens conforme são gerados (usando queue.Queue thread-safe)
        tokens = []
        while True:
            token = await loop.run_in_executor(None, callback.token_queue.get)
            if token is None:  # Fim da geração
                break
            tokens.append(token)

        # Aguardar conclusão da chain
        full_response = await task

        return tokens, full_response

    def process_prompt_sync(
        self,
        user_prompt: str,
        context: str = "General knowledge task"
    ) -> str:
        """
        Processa um prompt de forma síncrona (sem streaming).

        Args:
            user_prompt: Prompt do usuário
            context: Contexto adicional para o prompt

        Returns:
            Resposta completa do modelo
        """
        chain = self.create_chain()
        response = chain.run(user_input=user_prompt, context=context)
        return response


class EnhancedArenaOrchestrator(ArenaOrchestrator):
    """
    Versão aprimorada do orquestrador com funcionalidades extras.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Template ainda mais sofisticado com análise em múltiplas etapas
        self.enhanced_template = PromptTemplate(
            input_variables=["user_input", "context", "strategy"],
            template="""You are a top-tier AI assistant in a competitive environment.

Context: {context}
Strategy: {strategy}

Task: {user_input}

Instructions:
1. Analyze the question carefully
2. Structure your response logically
3. Provide concrete examples when relevant
4. Be concise yet comprehensive
5. Demonstrate expertise

Response:"""
        )

    def create_enhanced_chain(
        self,
        callback_handler: Optional[BaseCallbackHandler] = None,
        strategy: str = "accuracy_focused"
    ):
        """
        Cria uma chain aprimorada com estratégias customizáveis.

        Args:
            callback_handler: Callback para processar tokens
            strategy: Estratégia de resposta (accuracy_focused, speed_focused, detailed)

        Returns:
            LLMChain configurada
        """
        callbacks = []
        if callback_handler:
            callbacks.append(callback_handler)

        # Ajustar parâmetros baseado na estratégia
        temp = self.temperature
        max_tok = self.max_tokens

        if strategy == "speed_focused":
            temp = 0.3
            max_tok = 200
        elif strategy == "detailed":
            temp = 0.8
            max_tok = 1000
        elif strategy == "accuracy_focused":
            temp = 0.5
            max_tok = 500

        llm = Ollama(
            base_url=f"http://{self.ollama_host}:11434",
            model=self.ollama_model,
            temperature=temp,
            num_predict=max_tok,
            callbacks=callbacks,
        )

        chain = LLMChain(
            llm=llm,
            prompt=self.enhanced_template,
            verbose=False,
        )

        return chain


# Função helper para facilitar uso
def create_orchestrator(
    ollama_host: str = "localhost",
    ollama_model: str = "llama3.2",
    enhanced: bool = False,
    **kwargs
) -> ArenaOrchestrator:
    """
    Factory function para criar um orquestrador.

    Args:
        ollama_host: Host do Ollama
        ollama_model: Modelo a usar
        enhanced: Se True, usa EnhancedArenaOrchestrator
        **kwargs: Argumentos adicionais para o orquestrador

    Returns:
        Instância do orquestrador
    """
    if enhanced:
        return EnhancedArenaOrchestrator(
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            **kwargs
        )
    return ArenaOrchestrator(
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        **kwargs
    )


if __name__ == "__main__":
    # Exemplo de uso
    async def test_orchestrator():
        orchestrator = create_orchestrator(
            ollama_model="llama3.2",
            temperature=0.7
        )

        tokens, response = await orchestrator.process_prompt_streaming(
            user_prompt="What is the capital of France?",
            context="Geography quiz"
        )

        print(f"Tokens gerados: {len(tokens)}")
        print(f"Resposta: {response}")

    asyncio.run(test_orchestrator())
