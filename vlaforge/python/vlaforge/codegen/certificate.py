"""Render the compiler legality certificate as allocation-free C++ tables."""

from __future__ import annotations

from vlaforge.compiler import CompilationCertificate


def render_optimization_certificate_header(
    certificate: CompilationCertificate,
    *,
    namespace: str,
) -> str:
    enabled = tuple(item for item in certificate.caches if item.enabled)
    tables = []
    for cache in enabled:
        dependencies = []
        for item in cache.dependencies:
            kind = "0u" if item.kind == "epoch" else "1u"
            max_age = (
                "UINT64_MAX"
                if item.max_age_ns is None
                else f"{item.max_age_ns}u"
            )
            max_versions = (
                "UINT64_MAX"
                if item.max_versions is None
                else f"{item.max_versions}u"
            )
            dependencies.append(
                "  {"
                f"{kind}, {item.subject_id}u, {max_age}, {max_versions}"
                "},"
            )
        tables.append(
            f"""constexpr CertifiedTemporalDependency
    kCacheTask{cache.task_id}Dependencies[] = {{
{chr(10).join(dependencies)}
}};
constexpr std::size_t kCacheTask{cache.task_id}DependencyCount =
    sizeof(kCacheTask{cache.task_id}Dependencies) /
    sizeof(kCacheTask{cache.task_id}Dependencies[0]);"""
        )
    task_ids = ", ".join(f"{item.task_id}u" for item in enabled)
    return f"""#ifndef VLAFORGE_GENERATED_OPTIMIZATION_CERTIFICATE_H_
#define VLAFORGE_GENERATED_OPTIMIZATION_CERTIFICATE_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace {namespace} {{

struct CertifiedTemporalDependency {{
  std::uint32_t kind;
  std::uint32_t subject_id;
  std::uint64_t max_age_ns;
  std::uint64_t max_versions;
}};

constexpr char kCompilerProfile[] = "{certificate.profile.value}";
constexpr bool kCompilerProfileTestOnly =
    {"true" if certificate.test_only else "false"};
constexpr char kCompilationCertificateDigest[] =
    "{certificate.digest()}";
constexpr std::array<std::uint32_t, {len(enabled)}>
    kCertifiedCacheTaskIds{{{{{task_ids}}}}};
constexpr std::size_t kCertifiedCacheTaskCount = {len(enabled)}u;

{chr(10).join(tables)}

}}  // namespace {namespace}

#endif  // VLAFORGE_GENERATED_OPTIMIZATION_CERTIFICATE_H_
"""
