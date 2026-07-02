using Xunit;
using Calc;

// PROTECTED ORACLE (ADR-076): this test ships in the seed workspace and the
// cell-worker has no write capability over it. The model's job is to write
// Calc/StringOps.cs so that `dotnet test` here goes green. A verdict of
// `accepted` therefore means the box produced C# that compiles AND satisfies a
// contract it could not itself author.
public class SlugifyTests
{
    [Theory]
    [InlineData("Hello World!", "hello-world")]
    [InlineData("  Rust & C#  ", "rust-c")]
    [InlineData("Already-slugged", "already-slugged")]
    [InlineData("Multiple   spaces", "multiple-spaces")]
    [InlineData("__Trim__me__", "trim-me")]
    [InlineData("MiXeD CaSe 123", "mixed-case-123")]
    public void Slugifies(string input, string expected)
        => Assert.Equal(expected, StringOps.Slugify(input));
}
