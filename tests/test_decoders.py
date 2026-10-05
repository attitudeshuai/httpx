from __future__ import annotations

import io
import typing
import zlib

import chardet
import pytest
import zstandard as zstd

import httpx


def test_deflate():
    """
    Deflate encoding may use either 'zlib' or 'deflate' in the wild.

    https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib#answer-22311297
    """
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed_body = compressor.compress(body) + compressor.flush()

    headers = [(b"Content-Encoding", b"deflate")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zlib():
    """
    Deflate encoding may use either 'zlib' or 'deflate' in the wild.

    https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib#answer-22311297
    """
    body = b"test 123"
    compressed_body = zlib.compress(body)

    headers = [(b"Content-Encoding", b"deflate")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_gzip():
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = compressor.compress(body) + compressor.flush()

    headers = [(b"Content-Encoding", b"gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_brotli():
    body = b"test 123"
    compressed_body = b"\x8b\x03\x80test 123\x03"

    headers = [(b"Content-Encoding", b"br")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd():
    body = b"test 123"
    compressed_body = zstd.compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd_decoding_error():
    compressed_body = "this_is_not_zstd_compressed_data"

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200,
            headers=headers,
            content=compressed_body,
        )


def test_zstd_empty():
    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=b"")
    assert response.content == b""


def test_zstd_truncated():
    body = b"test 123"
    compressed_body = zstd.compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200,
            headers=headers,
            content=compressed_body[1:3],
        )


def test_zstd_multiframe():
    # test inspired by urllib3 test suite
    data = (
        # Zstandard frame
        zstd.compress(b"foo")
        # skippable frame (must be ignored)
        + bytes.fromhex(
            "50 2A 4D 18"  # Magic_Number (little-endian)
            "07 00 00 00"  # Frame_Size (little-endian)
            "00 00 00 00 00 00 00"  # User_Data
        )
        # Zstandard frame
        + zstd.compress(b"bar")
    )
    compressed_body = io.BytesIO(data)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=compressed_body)
    response.read()
    assert response.content == b"foobar"


def test_multi():
    body = b"test 123"

    deflate_compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed_body = deflate_compressor.compress(body) + deflate_compressor.flush()

    gzip_compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = (
        gzip_compressor.compress(compressed_body) + gzip_compressor.flush()
    )

    headers = [(b"Content-Encoding", b"deflate, gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_multi_with_identity():
    body = b"test 123"
    compressed_body = b"\x8b\x03\x80test 123\x03"

    headers = [(b"Content-Encoding", b"br, identity")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body

    headers = [(b"Content-Encoding", b"identity, br")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


@pytest.mark.anyio
async def test_streaming():
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)

    async def compress(body: bytes) -> typing.AsyncIterator[bytes]:
        yield compressor.compress(body)
        yield compressor.flush()

    headers = [(b"Content-Encoding", b"gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compress(body),
    )
    assert not hasattr(response, "body")
    assert await response.aread() == body


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br", b"identity"))
def test_empty_content(header_value):
    headers = [(b"Content-Encoding", header_value)]
    response = httpx.Response(
        200,
        headers=headers,
        content=b"",
    )
    assert response.content == b""


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br", b"identity"))
def test_decoders_empty_cases(header_value):
    headers = [(b"Content-Encoding", header_value)]
    response = httpx.Response(content=b"", status_code=200, headers=headers)
    assert response.read() == b""


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br"))
def test_decoding_errors(header_value):
    headers = [(b"Content-Encoding", header_value)]
    compressed_body = b"invalid"
    with pytest.raises(httpx.DecodingError):
        request = httpx.Request("GET", "https://example.org")
        httpx.Response(200, headers=headers, content=compressed_body, request=request)

    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers=headers, content=compressed_body)


@pytest.mark.parametrize(
    ["data", "encoding"],
    [
        ((b"Hello,", b" world!"), "ascii"),
        ((b"\xe3\x83", b"\x88\xe3\x83\xa9", b"\xe3", b"\x83\x99\xe3\x83\xab"), "utf-8"),
        ((b"Euro character: \x88! abcdefghijklmnopqrstuvwxyz", b""), "cp1252"),
        ((b"Accented: \xd6sterreich abcdefghijklmnopqrstuvwxyz", b""), "iso-8859-1"),
    ],
)
@pytest.mark.anyio
async def test_text_decoder_with_autodetect(data, encoding):
    async def iterator() -> typing.AsyncIterator[bytes]:
        nonlocal data
        for chunk in data:
            yield chunk

    def autodetect(content):
        return chardet.detect(content).get("encoding")

    # Accessing `.text` on a read response.
    response = httpx.Response(200, content=iterator(), default_encoding=autodetect)
    await response.aread()
    assert response.text == (b"".join(data)).decode(encoding)

    # Streaming `.aiter_text` iteratively.
    # Note that if we streamed the text *without* having read it first, then
    # we won't get a `charset_normalizer` guess, and will instead always rely
    # on utf-8 if no charset is specified.
    text = "".join([part async for part in response.aiter_text()])
    assert text == (b"".join(data)).decode(encoding)


@pytest.mark.anyio
async def test_text_decoder_known_encoding():
    async def iterator() -> typing.AsyncIterator[bytes]:
        yield b"\x83g"
        yield b"\x83"
        yield b"\x89\x83x\x83\x8b"

    response = httpx.Response(
        200,
        headers=[(b"Content-Type", b"text/html; charset=shift-jis")],
        content=iterator(),
    )

    await response.aread()
    assert "".join(response.text) == "トラベル"


def test_text_decoder_empty_cases():
    response = httpx.Response(200, content=b"")
    assert response.text == ""

    response = httpx.Response(200, content=[b""])
    response.read()
    assert response.text == ""


@pytest.mark.parametrize(
    ["data", "expected"],
    [((b"Hello,", b" world!"), ["Hello,", " world!"])],
)
def test_streaming_text_decoder(
    data: typing.Iterable[bytes], expected: list[str]
) -> None:
    response = httpx.Response(200, content=iter(data))
    assert list(response.iter_text()) == expected


def test_line_decoder_nl():
    response = httpx.Response(200, content=[b""])
    assert list(response.iter_lines()) == []

    response = httpx.Response(200, content=[b"", b"a\n\nb\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(
        200, content=[b"", b"12345\n", b"foo ", b"bar ", b"baz\n"]
    )
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_line_decoder_cr():
    response = httpx.Response(200, content=[b"", b"a\r\rb\rc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r\rb\rc\r"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(
        200, content=[b"", b"12345\r", b"foo ", b"bar ", b"baz\r"]
    )
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_line_decoder_crnl():
    response = httpx.Response(200, content=[b"", b"a\r\n\r\nb\r\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r\n\r\nb\r\nc\r\n"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r", b"\n\r\nb\r\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(200, content=[b"", b"12345\r\n", b"foo bar baz\r\n"])
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_invalid_content_encoding_header():
    headers = [(b"Content-Encoding", b"invalid-header")]
    body = b"test 123"

    response = httpx.Response(
        200,
        headers=headers,
        content=body,
    )
    assert response.content == body


# ---------------------------------------------------------------------------
# Configurable content-encoding negotiation and decoder chains.
# ---------------------------------------------------------------------------


def gzip_compress(data: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    return compressor.compress(data) + compressor.flush()


def gzip_wire_chunks(data: bytes, chunk_size: int = 3) -> list[bytes]:
    compressed = gzip_compress(data)
    return [
        compressed[i : i + chunk_size]
        for i in range(0, len(compressed), chunk_size)
    ]


class ChunkedByteStream(httpx.SyncByteStream):
    """
    A byte stream which yields fixed chunks and records close calls.
    """

    def __init__(self, chunks: typing.Iterable[bytes]) -> None:
        self._chunks = list(chunks)
        self.is_closed = False

    def __iter__(self) -> typing.Iterator[bytes]:
        yield from self._chunks

    def close(self) -> None:
        self.is_closed = True


def test_default_accept_encoding_is_byte_identical():
    # Unconfigured clients must declare exactly what httpx declared before.
    client = httpx.Client()
    assert client.headers["Accept-Encoding"] == "gzip, deflate, br, zstd"
    request = client.build_request("GET", "https://example.org")
    assert request.headers["Accept-Encoding"] == "gzip, deflate, br, zstd"


def test_client_limited_accept_encoding():
    client = httpx.Client(accept_encoding=["gzip", "deflate"])
    assert client.headers["Accept-Encoding"] == "gzip, deflate"

    # Tokens are stripped, lowercased and deduplicated.
    client = httpx.Client(accept_encoding=[" GZIP ", "gzip", "Deflate"])
    assert client.headers["Accept-Encoding"] == "gzip, deflate"


def test_client_disabled_accept_encoding():
    client = httpx.Client(accept_encoding=None)
    assert "accept-encoding" not in client.headers
    request = client.build_request("GET", "https://example.org")
    assert "accept-encoding" not in request.headers


def test_client_string_accept_encoding_used_verbatim():
    client = httpx.Client(accept_encoding="gzip;q=0.8, br;q=0.5")
    assert client.headers["Accept-Encoding"] == "gzip;q=0.8, br;q=0.5"


def test_per_request_accept_encoding_override():
    client = httpx.Client()

    request = client.build_request(
        "GET", "https://example.org", accept_encoding=["br"]
    )
    assert request.headers["Accept-Encoding"] == "br"

    request = client.build_request(
        "GET", "https://example.org", accept_encoding=None
    )
    assert "accept-encoding" not in request.headers

    request = client.build_request(
        "GET", "https://example.org", accept_encoding="deflate;q=1.0"
    )
    assert request.headers["Accept-Encoding"] == "deflate;q=1.0"


def test_accept_encoding_declaration_sent_to_server():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept-encoding"] = request.headers.get("accept-encoding", "")
        return httpx.Response(200, content=b"")

    # Client-level configuration.
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        accept_encoding=["gzip", "deflate"],
    )
    client.get("https://example.org")
    assert seen["accept-encoding"] == "gzip, deflate"

    # Per-request configuration through `request()`.
    client = httpx.Client(transport=httpx.MockTransport(handler))
    client.request(
        "GET", "https://example.org", accept_encoding=["br"]
    )
    assert seen["accept-encoding"] == "br"


def test_invalid_encoding_configuration_raises():
    # An acceptable set may only name encodings with an installed backend.
    with pytest.raises(ValueError):
        httpx.Client(decodable_encodings=["bogus"])

    with pytest.raises(ValueError):
        httpx.Client(decodable_encodings=[""])

    with pytest.raises(TypeError):
        httpx.Client(unsupported_encoding_policy="raise")  # type: ignore

    client = httpx.Client()
    with pytest.raises(ValueError):
        client.build_request(
            "GET", "https://example.org", decodable_encodings=["bogus"]
        )


BROTLI_BODY = b"\x8b\x03\x80test 123\x03"


def test_excluded_encoding_passes_through_by_default():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"br")],
            stream=httpx.ByteStream(BROTLI_BODY),
        )

    # `br` has a backend normally, but is outside this client's acceptable set.
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        decodable_encodings=["gzip"],
    )
    response = client.get("https://example.org")
    assert response.content == BROTLI_BODY


def test_excluded_encoding_raises_when_strict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"br")],
            stream=httpx.ByteStream(BROTLI_BODY),
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        decodable_encodings=["gzip"],
        unsupported_encoding_policy=httpx.UnsupportedEncodingPolicy.RAISE,
    )
    with pytest.raises(httpx.UnsupportedEncodingError) as exc_info:
        client.get("https://example.org")
    assert exc_info.value.encoding == "br"


def test_unknown_encoding_passes_through_by_default():
    # Duplicated from the standard behaviour, explicit for the contrast below.
    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"bogus")],
        content=b"some body",
    )
    assert response.content == b"some body"


def test_unknown_encoding_raises_when_strict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"bogus")],
            stream=httpx.ByteStream(b"some body"),
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        unsupported_encoding_policy=httpx.UnsupportedEncodingPolicy.RAISE,
    )
    with pytest.raises(httpx.UnsupportedEncodingError) as exc_info:
        client.get("https://example.org")
    exc = exc_info.value
    assert exc.encoding == "bogus"
    assert exc.layer == 0
    assert isinstance(exc, httpx.DecodingError)


def test_unknown_layer_index_is_counted_from_outside():
    # 'Content-Encoding: gzip, bogus': bogus is the outermost layer (0),
    # gzip the inner layer (1).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"gzip, bogus")],
            stream=httpx.ByteStream(b""),
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        unsupported_encoding_policy=httpx.UnsupportedEncodingPolicy.RAISE,
    )
    with pytest.raises(httpx.UnsupportedEncodingError) as exc_info:
        client.get("https://example.org")
    assert exc_info.value.encoding == "bogus"
    assert exc_info.value.layer == 0


def test_missing_backend_passes_through_by_default(monkeypatch):
    import httpx._decoders as decoders_module

    monkeypatch.delitem(
        decoders_module.SUPPORTED_DECODERS, "br", raising=False
    )

    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"br")],
        content=BROTLI_BODY,
    )
    assert response.content == BROTLI_BODY


def test_missing_backend_raises_when_strict(monkeypatch):
    import httpx._decoders as decoders_module

    monkeypatch.delitem(
        decoders_module.SUPPORTED_DECODERS, "br", raising=False
    )

    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"br")],
        stream=httpx.ByteStream(BROTLI_BODY),
    )
    response.unsupported_encoding_policy = (
        httpx.UnsupportedEncodingPolicy.RAISE
    )
    with pytest.raises(httpx.UnsupportedEncodingError) as exc_info:
        response.read()
    assert exc_info.value.encoding == "br"


def test_multi_layer_failure_is_attributed():
    # Wire bytes are gzip(b"this is definitely not gzip encoded"): the outer
    # gzip layer decodes cleanly, the inner gzip layer fails.
    inner_raw = b"this is definitely not gzip encoded"
    wire = gzip_compress(inner_raw)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"gzip, gzip")],
            stream=httpx.ByteStream(wire),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.DecodingError) as exc_info:
        client.get("https://example.org")
    exc = exc_info.value
    assert exc.encoding == "gzip"
    assert exc.layer == 1  # outer gzip layer 0, inner gzip layer 1
    assert exc.partial_content == inner_raw
    assert "layer 2 of 2" in str(exc)


def test_outer_layer_failure_is_attributed():
    # 'Content-Encoding: br, gzip': gzip is the outermost layer (0).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"br, gzip")],
            stream=httpx.ByteStream(b"not gzip at all"),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.DecodingError) as exc_info:
        client.get("https://example.org")
    exc = exc_info.value
    assert exc.encoding == "gzip"
    assert exc.layer == 0
    assert exc.partial_content == b""


def test_zstd_truncated_failure_is_attributed():
    compressed_body = zstd.compress(b"test 123")
    with pytest.raises(httpx.DecodingError) as exc_info:
        httpx.Response(
            200,
            headers=[("Content-Encoding", b"zstd")],
            content=compressed_body[1:3],
        )
    exc = exc_info.value
    assert exc.encoding == "zstd"
    assert exc.layer == 0
    assert exc.partial_content == b""


def test_streaming_failure_preserves_decoded_prefix():
    # Find a prefix of the gzip stream which decodes to exactly b"hello ".
    wire = gzip_compress(b"hello world, this is a test body!")
    split = next(
        i
        for i in range(2, len(wire))
        if zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(wire[:i])
        == b"hello "
    )

    def content() -> typing.Iterator[bytes]:
        yield wire[:split]
        yield b"\xff\xff\xff garbage bytes"

    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"gzip")],
        content=content(),
    )
    with pytest.raises(httpx.DecodingError) as exc_info:
        response.read()
    exc = exc_info.value
    assert exc.encoding == "gzip"
    assert exc.layer == 0
    assert exc.partial_content == b"hello "


def test_early_generator_close_is_deterministic():
    def content() -> typing.Iterator[bytes]:
        yield gzip_compress(b"hello ")
        yield b"not-a-gzip-stream"  # must never be decoded or flushed

    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"gzip")],
        content=content(),
    )
    iterator = response.iter_bytes()
    assert next(iterator) == b"hello "
    # Closing the iterator early must not flush the chain and not raise
    # a decoding error; the response is simply closed.
    iterator.close()
    assert response.is_closed

    with pytest.raises(httpx.StreamClosed):
        response.read()


def test_close_does_not_run_decoder_flush():
    raw_stream = ChunkedByteStream(gzip_wire_chunks(b"hello world"))
    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"gzip")],
        stream=raw_stream,
    )
    response.close()
    assert response.is_closed
    assert raw_stream.is_closed


def test_decoding_error_is_separate_from_transport_error():
    # Decoding failure...
    def decode_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"gzip")],
            stream=httpx.ByteStream(b"not gzip"),
        )

    client = httpx.Client(transport=httpx.MockTransport(decode_failure))
    with pytest.raises(httpx.DecodingError) as exc_info:
        client.get("https://example.org")
    assert not isinstance(exc_info.value, httpx.TransportError)

    # ...versus a network read failure.
    def network_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset by peer")

    client = httpx.Client(transport=httpx.MockTransport(network_failure))
    with pytest.raises(httpx.ReadError) as exc_info:
        client.get("https://example.org")
    assert isinstance(exc_info.value, httpx.TransportError)
    assert not isinstance(exc_info.value, httpx.DecodingError)


def test_per_request_policy_overrides_client_default():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"bogus")],
            stream=httpx.ByteStream(b"body"),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    # A strict request on a default (passthrough) client fails explicitly.
    request = client.build_request(
        "GET",
        "https://example.org",
        unsupported_encoding_policy=httpx.UnsupportedEncodingPolicy.RAISE,
    )
    with pytest.raises(httpx.UnsupportedEncodingError):
        client.send(request)

    # An ordinary request on the same client still passes through.
    response = client.get("https://example.org")
    assert response.content == b"body"


def test_per_request_decodable_subset():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"br")],
            stream=httpx.ByteStream(BROTLI_BODY),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    request = client.build_request(
        "GET",
        "https://example.org",
        decodable_encodings=["gzip"],
    )
    response = client.send(request)
    # br is outside the per-request acceptable set -> default pass through.
    assert response.content == BROTLI_BODY


def test_content_encodings_property():
    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"gzip, deflate")],
        content=b"",
    )
    assert response.content_encodings == ("gzip", "deflate")

    response = httpx.Response(200, content=b"")
    assert response.content_encodings == ()


def test_response_decoder_state_isolation():
    body_a = b"Hello, world!" * 20
    body_b = b"Different response body content." * 20
    chunks_a = gzip_wire_chunks(body_a)
    chunks_b = gzip_wire_chunks(body_b)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            chunks = chunks_a
        else:
            chunks = chunks_b
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"gzip")],
            stream=ChunkedByteStream(chunks),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with client.stream("GET", "https://example.org/a") as response_a:
        with client.stream("GET", "https://example.org/b") as response_b:
            iter_a = response_a.iter_bytes(5)
            iter_b = response_b.iter_bytes(5)
            out_a: list[bytes] = []
            out_b: list[bytes] = []
            while True:
                active = False
                try:
                    out_a.append(next(iter_a))
                    active = True
                except StopIteration:
                    pass
                try:
                    out_b.append(next(iter_b))
                    active = True
                except StopIteration:
                    pass
                if not active:
                    break

    assert b"".join(out_a) == body_a
    assert b"".join(out_b) == body_b
    # Each response carries its own byte counter, counting the raw wire bytes.
    assert response_a.num_bytes_downloaded == len(b"".join(chunks_a))
    assert response_b.num_bytes_downloaded == len(b"".join(chunks_b))
    assert (
        response_a.num_bytes_downloaded != response_b.num_bytes_downloaded
    )


@pytest.mark.anyio
async def test_async_early_close_is_deterministic():
    async def content() -> typing.AsyncIterator[bytes]:
        yield gzip_compress(b"hello ")
        yield b"not gzip"  # must never be decoded or flushed

    response = httpx.Response(
        200,
        headers=[("Content-Encoding", b"gzip")],
        content=content(),
    )
    iterator = response.aiter_bytes()
    assert await iterator.__anext__() == b"hello "
    # Early aclose must not flush and not raise a decoding error.
    await iterator.aclose()
    assert response.is_closed


@pytest.mark.anyio
async def test_async_strict_policy_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Encoding", b"bogus")],
            stream=httpx.ByteStream(b"body"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        unsupported_encoding_policy=httpx.UnsupportedEncodingPolicy.RAISE,
    ) as client:
        with pytest.raises(httpx.UnsupportedEncodingError) as exc_info:
            await client.get("https://example.org")
        assert exc_info.value.encoding == "bogus"
