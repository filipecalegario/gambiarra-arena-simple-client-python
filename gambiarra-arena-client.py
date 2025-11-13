#!/usr/bin/env python3
import asyncio
import json
import time
import uuid

import websockets

# ============================================
# CONFIGURAÇÕES - Edite aqui conforme necessário
# ============================================
URL = "ws://192.168.0.212:3000/ws"  # URL do servidor WebSocket
PIN = "937414"  # PIN da sessão
PARTICIPANT_ID = "meuId"  # Se None, será gerado um UUID automaticamente
NICKNAME = "Python Mock"  # Apelido do participante
TOKEN_DELAY = 0.05  # Delay em segundos entre tokens mockados
# ============================================


async def handle_challenge(ws, challenge: dict):
    round_id = challenge.get("round")
    prompt = challenge.get("prompt", "")

    print("\n=== Novo desafio recebido ===")
    print(f"round: {round_id}")
    print(f"prompt: {prompt}")
    print("=============================\n")

    full_response = (
        f"[MOCK] Esta é uma resposta mockada do participante "
        f"'{NICKNAME}' para o prompt: {prompt}"
    )

    tokens = full_response.split()
    print(f"Tokens mockados gerados ({len(tokens)} tokens): {tokens}")
    total_tokens = len(tokens)

    start_time = time.perf_counter()
    first_token_time = None

    print("Enviando tokens mockados...")
    for seq, token in enumerate(tokens):
        if first_token_time is None:
            first_token_time = time.perf_counter()

        token_msg = {
            "type": "token",
            "round": round_id,
            "participant_id": PARTICIPANT_ID,
            "seq": seq,
            "content": token,
        }

        await ws.send(json.dumps(token_msg))
        await asyncio.sleep(TOKEN_DELAY)

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    latency_first_token_ms = int((first_token_time - start_time) * 1000)
    duration_ms = int((end_time - start_time) * 1000)

    complete_msg = {
        "type": "complete",
        "round": round_id,
        "participant_id": PARTICIPANT_ID,
        "tokens": total_tokens,  # ← métricas no nível raiz
        "latency_ms_first_token": latency_first_token_ms,
        "duration_ms": duration_ms,
        # tpsAvg removido - não existe no schema
    }

    print("Enviando mensagem de conclusão com métricas:")
    print(json.dumps(complete_msg, indent=2, ensure_ascii=False))

    await ws.send(json.dumps(complete_msg))


async def client_loop():
    """
    Faz a conexão, registra o participante e entra no loop
    para receber mensagens do servidor.
    """

    print(f"Conectando a {URL} ...")

    async with websockets.connect(URL) as ws:
        print("✅ Conectado ao servidor WebSocket.")

        # Mensagem de registro inicial (Client → Server)
        register_msg = {
            "type": "register",
            "participant_id": PARTICIPANT_ID,
            "nickname": NICKNAME,
            "pin": PIN,
            "runner": "mock",
            "model": "MOCK-1.0",
        }

        print("Enviando mensagem de registro:")
        print(json.dumps(register_msg, indent=2, ensure_ascii=False))
        await ws.send(json.dumps(register_msg))

        # Loop principal de recebimento de mensagens
        while True:
            try:
                raw_msg = await ws.recv()
            except websockets.ConnectionClosed as e:
                print(f"Conexão fechada pelo servidor: {e.code} - {e.reason}")
                break

            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                print(f"Mensagem não é JSON válido: {raw_msg}")
                continue

            msg_type = data.get("type")

            if msg_type == "challenge":
                # Mensagem de desafio: gera resposta mockada
                await handle_challenge(ws, data)

            elif msg_type == "heartbeat":
                # Heartbeat do servidor — aqui só logamos
                print("💓 Heartbeat recebido do servidor.")

            else:
                print("📩 Mensagem desconhecida recebida:")
                print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    try:
        asyncio.run(client_loop())
    except KeyboardInterrupt:
        print("\nEncerrando cliente por KeyboardInterrupt.")


if __name__ == "__main__":
    main()
