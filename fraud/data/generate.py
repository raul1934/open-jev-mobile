#!/usr/bin/env python3
"""Synthetic Portuguese (Brazil) training data for scam-message detection.

Every message here is invented: templates for the common scam families in
Brazil, and legitimate messages that share their vocabulary (real bank
notices, delivery updates, verification codes, promotions, family chat), so
the model cannot just learn "mentions a bank = scam".

The evaluation set (test.jsonl) is written by hand with different wording and
is never generated here. Real messages the user labels go in real.jsonl and
are used by train.py for both training and evaluation.

    python fraud/data/generate.py            # writes fraud/data/train.jsonl
"""
import json
import random
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train.jsonl"
rng = random.Random(20260924)

NAMES = ["Ana", "Bruno", "Carla", "Diego", "Eduarda", "Felipe", "Gabriela", "Henrique", "Isabela", "João",
         "Larissa", "Marcos", "Natália", "Otávio", "Paula", "Rafael", "Sabrina", "Tiago", "Vanessa", "Wagner"]
RELATIVES = ["mãe", "pai", "filho", "filha", "tia", "vó", "amor", "mano", "prima", "tio"]
BANKS = ["Banco do Brasil", "Caixa", "Itaú", "Bradesco", "Santander", "Nubank", "Inter", "C6 Bank", "BB", "Sicredi"]
STORES = ["Magalu", "Americanas", "Casas Bahia", "Mercado Livre", "Shopee", "Amazon", "Renner", "Riachuelo"]
CARRIERS = ["Correios", "Loggi", "Jadlog", "Total Express", "Sequoia"]
SCAM_DOMAINS = ["bit.ly/{c}", "tinyurl.com/{c}", "{b}-seguranca.com", "atendimento-{b}.net", "{b}.regularize-agora.xyz",
                "correios-taxa.{t}", "rastreio-{c}.shop", "premio-{c}.site", "gov-{c}.online", "wa.me/55119{d}",
                "{b}-cliente.top", "desbloqueio{d}.com", "{b}app.info", "cadastro-{c}.click", "l.ead.me/{c}"]
REAL_DOMAINS = ["bb.com.br", "caixa.gov.br", "itau.com.br", "bradesco.com.br", "nubank.com.br", "correios.com.br",
                "mercadolivre.com.br", "magazineluiza.com.br", "gov.br"]
TLDS = ["xyz", "top", "shop", "site", "online", "click", "info", "live"]


def code(n=6):
    return "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(n))


def digits(n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def money():
    return rng.choice(["R$ {}".format(rng.choice([49, 89, 150, 299, 480, 750, 1.200, 1.890, 2.500, 3.450, 4.999])),
                       "R${},{:02d}".format(rng.randint(10, 999), rng.randint(0, 99)),
                       "R$ {}.{:03d},00".format(rng.randint(1, 9), rng.randint(0, 999))])


def scam_link():
    bank = rng.choice(["bb", "caixa", "itau", "bradesco", "nubank", "santander", "inter", "serasa", "detran", "receita"])
    return "https://" + rng.choice(SCAM_DOMAINS).format(b=bank, c=code(), d=digits(4), t=rng.choice(TLDS))


def real_link():
    return rng.choice(["", "https://www.", "www."]) + rng.choice(REAL_DOMAINS)


def date():
    return "{:02d}/{:02d}".format(rng.randint(1, 28), rng.randint(1, 12))


def hour():
    return "{:02d}:{:02d}".format(rng.randint(6, 23), rng.choice([0, 15, 30, 45, 10, 42]))


def fill(t):
    return t.format(
        nome=rng.choice(NAMES), parente=rng.choice(RELATIVES), banco=rng.choice(BANKS), loja=rng.choice(STORES),
        transp=rng.choice(CARRIERS), valor=money(), link=scam_link(), link_real=real_link(), data=date(), hora=hour(),
        cod=digits(6), final=digits(4), pedido=digits(8), prot=digits(10), tel="(11) 9" + digits(4) + "-" + digits(4),
    )


SCAMS = {
    "banco_falso": [
        "{banco}: identificamos uma compra de {valor} no seu cartão final {final}. Se não reconhece, acesse {link} imediatamente.",
        "Seu cartão {banco} foi BLOQUEADO por suspeita de fraude. Regularize em até 24h: {link}",
        "Prezado cliente, sua conta {banco} será suspensa hoje. Atualize seus dados de segurança em {link}",
        "{banco} informa: seu token expirou. Para evitar o bloqueio da conta acesse {link} e informe sua senha.",
        "ALERTA {banco}: tentativa de acesso ao seu app em outro celular. Nao foi voce? Cancele agora {link}",
        "Olá, aqui é da central de segurança do {banco}. Detectamos movimentação suspeita, me confirme sua senha do cartão para cancelarmos.",
        "Cliente {banco}, você possui {valor} em pontos que vencem hoje. Resgate em {link}",
        "{banco}: seu cadastro está desatualizado e o PIX será desativado. Regularize: {link}",
        "Compra aprovada {valor} {loja} cartão final {final}. Não reconhece? Ligue {tel} para cancelar.",
    ],
    "parente_numero_novo": [
        "Oi {parente}, esse é meu número novo, salva aí. O outro celular caiu na água",
        "{parente} troquei de número, pode apagar o antigo. Me faz um favor rapidinho?",
        "Oi {parente}! Tô com um problema no app do banco, consegue fazer um pix de {valor} pra mim? Te devolvo amanhã",
        "{parente}, é urgente, preciso pagar uma conta hoje e meu banco travou. Me empresta {valor}? Mando a chave pix",
        "Oii {parente} tudo bem? Mudei de numero. To precisando de uma ajuda, pode falar?",
        "{parente} sou eu, {nome}. Perdi meu celular, esse é o novo. Consegue me transferir {valor}? É emergência",
        "Mãe, salva esse número novo. Preciso te pedir uma coisa mas não conta pro pai",
    ],
    "pix_errado": [
        "Oi, fiz um pix de {valor} pra você por engano, pode me devolver? Mando o comprovante",
        "Boa tarde, transferi {valor} errado pra sua conta, por favor devolva nessa chave: {tel}",
        "Olá {nome}, caiu um PIX de {valor} na sua conta por erro meu, consegue estornar pelo link {link}?",
    ],
    "entrega_taxa": [
        "{transp}: sua encomenda está retida na fiscalização. Pague a taxa de {valor} para liberar: {link}",
        "Seu pedido {pedido} não pôde ser entregue por endereço incompleto. Atualize em {link}",
        "{transp} informa: objeto aguardando pagamento de imposto de importação. Evite devolução: {link}",
        "Sua entrega foi reagendada. Confirme seus dados e pague o frete de {valor} em {link}",
        "Encomenda taxada! Regularize a taxa alfandegária até {data} ou o objeto será devolvido: {link}",
    ],
    "premio_sorteio": [
        "PARABÉNS! Seu número foi sorteado e você ganhou {valor} no sorteio {loja}. Resgate: {link}",
        "Você foi selecionado para receber um iPhone 16 grátis! Pague só o frete em {link}",
        "{loja} 30 anos: responda a pesquisa e ganhe um vale de {valor} {link}",
        "Seu CPF tem {valor} a receber do governo. Consulte agora em {link}",
        "Valores esquecidos: você tem {valor} para sacar. Solicite até hoje: {link}",
    ],
    "emprego_falso": [
        "Olá! Vaga home office, ganhe de R$ 300 a R$ 800 por dia curtindo vídeos. Chama no {link}",
        "Oportunidade: trabalhe 1h por dia avaliando produtos da {loja} e ganhe comissão. Fale com a recrutadora {link}",
        "Seu currículo foi aprovado! Para a vaga é necessário pagar {valor} de taxa de cadastro. Pix: {tel}",
        "Tarefa simples no Telegram, pagamento imediato via pix. Entre no grupo: {link}",
    ],
    "codigo_whatsapp": [
        "Oi, te mandei um código de 6 dígitos por SMS sem querer, pode me passar?",
        "Aqui é do suporte do WhatsApp. Para verificar sua conta, informe o código que você recebeu agora.",
        "Olá, estou atualizando o cadastro da sua consulta. Vai chegar um código no seu celular, me envia por favor",
        "{nome} aqui do grupo da escola, chegou um código aí? Mandei errado pro seu número, me passa?",
    ],
    "boleto_divida": [
        "Consta um débito em seu CPF no valor de {valor}. Evite negativação pagando o boleto: {link}",
        "SERASA: acordo com 90% de desconto disponível só hoje. Quite sua dívida em {link}",
        "Sua conta de luz está em atraso e será cortada amanhã. Pague agora: {link}",
        "Detran: você possui multa de {valor} vencida. Pague com desconto em {link}",
        "IPVA atrasado! Regularize seu veículo com 50% de desconto até {data}: {link}",
        "Receita Federal: seu CPF está irregular e será cancelado. Regularize em {link}",
    ],
    "suporte_falso": [
        "Aqui é do suporte da {loja}. Seu pedido {pedido} teve problema no pagamento, me passa os dados do cartão para refazer",
        "Sua conta do Instagram será excluída por violação. Recorra em {link}",
        "Netflix: não conseguimos processar seu pagamento. Atualize o cartão em {link}",
        "Técnico da operadora: seu chip será desativado. Instale este app para manter o número: {link}",
        "Olá, sou do setor financeiro da {loja}. Houve cobrança em duplicidade, para estornar informe senha e código do cartão.",
    ],
    "investimento": [
        "Invista {valor} e receba o dobro em 7 dias! Garantido. Chame no {link}",
        "Robô de investimentos com lucro de 5% ao dia. Vagas limitadas: {link}",
        "Ganhe dinheiro com cripto sem risco. Mentoria grátis no grupo {link}",
    ],
}

LEGIT = {
    "banco_real": [
        "{banco}: compra aprovada de {valor} em {loja}, cartão final {final}, {data} às {hora}.",
        "{banco}: você recebeu um Pix de {valor} de {nome}.",
        "{banco} nunca pede senha, token ou código por mensagem, ligação ou link. Em caso de dúvida, use o app oficial.",
        "Seu código de verificação {banco} é {cod}. Não compartilhe com ninguém.",
        "{banco}: sua fatura de {valor} vence em {data}. Pague pelo app.",
        "Transferência de {valor} realizada com sucesso para {nome} em {data}.",
        "{banco}: o limite do seu cartão foi aumentado. Confira no app.",
        "Seu cartão final {final} foi desbloqueado com sucesso.",
    ],
    "entrega_real": [
        "{loja}: seu pedido {pedido} saiu para entrega e chega hoje até as 21h.",
        "{transp}: objeto {pedido}BR entregue em {data} às {hora}.",
        "Seu pedido {pedido} foi enviado! Acompanhe pelo app da {loja}.",
        "O entregador está a caminho. Código de entrega: {cod}. Informe apenas ao receber o pacote.",
        "{loja}: seu pedido foi confirmado. Previsão de entrega: {data}.",
    ],
    "codigo_real": [
        "Seu código do WhatsApp: {cod}. Não compartilhe esse código.",
        "{cod} é o seu código de verificação. Válido por 10 minutos.",
        "Use o código {cod} para entrar na sua conta. Se não foi você, ignore esta mensagem.",
        "Código de acesso gov.br: {cod}",
    ],
    "familia_amigos": [
        "Oi {parente}, chego às {hora}, pode me buscar na rodoviária?",
        "{parente}, comprei pão, precisa de mais alguma coisa do mercado?",
        "Te fiz um pix de {valor} da pizza de ontem, confere aí",
        "Oi {nome}! Vai no aniversário da {nome} sábado?",
        "{parente} me manda a foto da receita do bolo por favor",
        "Chegou bem? Me avisa quando estiver em casa",
        "Amanhã a reunião da escola é às {hora}, não esquece",
        "{nome}, me devolve os {valor} do ingresso quando puder, sem pressa",
        "Mãe, meu celular tá quase sem bateria, te ligo mais tarde",
        "Salva meu número novo, é {nome}, perdi o chip antigo. Beijo!",
    ],
    "servicos": [
        "Lembrete: sua consulta com Dr. {nome} é dia {data} às {hora}. Responda 1 para confirmar.",
        "Sua conta de luz de {valor} vence em {data}. Pague pelo app ou nas lotéricas.",
        "Operadora: sua fatura de {valor} já está disponível no app.",
        "Seu agendamento no Poupatempo foi confirmado para {data} às {hora}.",
        "Detran: seu licenciamento está disponível para pagamento no site oficial {link_real}.",
        "Obrigado por abastecer! Você ganhou 50 pontos no programa de fidelidade.",
    ],
    "promocao_real": [
        "{loja}: até 40% off em eletrônicos neste fim de semana. Confira no app.",
        "Black Friday {loja}: frete grátis em todo o site. Aproveite!",
        "Seu cupom de 10% de desconto na {loja} é BEMVINDO10. Válido até {data}.",
        "Você tem produtos esquecidos no carrinho da {loja}. Finalize sua compra pelo app.",
    ],
    "trabalho": [
        "{nome}, a reunião foi remarcada para {hora}. Link no convite do calendário.",
        "Pessoal, o relatório precisa ser entregue até {data}.",
        "Oi {nome}, recebemos seu currículo e gostaríamos de agendar uma entrevista. Qual horário fica bom?",
        "O pagamento do seu salário foi creditado. Confira seu holerite no portal RH.",
    ],
}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def noise(s):
    """Writing-style variation seen in real messages."""
    r = rng.random()
    if r < 0.15:
        s = strip_accents(s)
    elif r < 0.22:
        s = s.upper()
    elif r < 0.35:
        s = s.lower()
    if rng.random() < 0.2:
        for a, b in [(" você", " vc"), (" por favor", " pfv"), (" para ", " pra "), (" que ", " q "), (" está", " tá")]:
            s = s.replace(a, b)
    if rng.random() < 0.15:
        s += " " + rng.choice(["🙏", "😊", "⚠️", "🚨", "❤️", "👍", "!!", "..."])
    return s


def main():
    rows = []
    for label, families in ((1, SCAMS), (0, LEGIT)):
        for family, templates in families.items():
            for t in templates:
                for _ in range(40):
                    rows.append({"text": noise(fill(t)), "label": label, "family": family, "source": "synthetic"})
    rng.shuffle(rows)
    # Drop exact duplicates produced by templates without slots.
    seen, unique = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    with OUT.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scams = sum(r["label"] for r in unique)
    print(f"wrote {len(unique)} messages ({scams} scams, {len(unique) - scams} legitimate) to {OUT.name}")


if __name__ == "__main__":
    main()
